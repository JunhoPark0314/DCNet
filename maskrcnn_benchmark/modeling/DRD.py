from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from maskrcnn_benchmark.utils.events import get_event_storage
 
class DenseRelationDistill(nn.Module):

    def __init__(self, indim, keydim, valdim, dense_sum=False, sigmoid_attn=False, no_padding=False, per_level_bn=False):
        super(DenseRelationDistill,self).__init__()
        #self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        #self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)

        if no_padding:
            padding = 0
            bn_dim = 196
        else:
            padding = (1,1)
            bn_dim = 256

        self.key_t = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
        self.value_t = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)

        self.sum = dense_sum
        self.sigmoid_attn = sigmoid_attn
        if self.sum:
            self.key_q0 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
            self.value_q0 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)
            self.key_q1 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
            self.value_q1 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)
            self.key_q2 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
            self.value_q2 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)
            self.key_q3 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
            self.value_q3 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)
            self.key_q4 = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=padding, stride=1)
            self.value_q4 = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=padding, stride=1)
            self.bnn0 = nn.BatchNorm2d(256)
            self.bnn1 = nn.BatchNorm2d(256)
            self.bnn2 = nn.BatchNorm2d(256)
            self.bnn3 = nn.BatchNorm2d(256)
            self.bnn4 = nn.BatchNorm2d(256)
            self.combine = nn.Conv2d(512,256,kernel_size=1,padding=0,stride=1)
        
        if self.sigmoid_attn:
            self.sigmoid_attn_bnn = nn.ModuleList([
                nn.BatchNorm1d(bn_dim) for _ in range(5)])

            if per_level_bn:
                self.attn_bnn = lambda x: self.sigmoid_attn_bnn[x]
            else:
                self.attn_bnn = lambda x: self.sigmoid_attn_bnn[0]

            self.temperature = 10
   
    def forward(self,features,attentions):
        features = list(features)
        if isinstance(attentions,dict):
            for i in range(len(attentions)):
                if i==0:
                    atten = attentions[i].unsqueeze(0)
                else:
                    atten = torch.cat((atten,attentions[i].unsqueeze(0)),dim=0)
            attentions = atten.cuda()
        output = []
        full_attention = defaultdict(list)
        h , w = attentions.shape[2:]
        ncls = attentions.shape[0]       
        key_t = self.key_t(attentions)   
        val_t = self.value_t(attentions) 

        att_h, att_w = val_t.shape[2:]
        for idx in range(len(features)):
            feature = features[idx]
            bs = feature.shape[0]       
            H , W = feature.shape[2:]
            feature = F.interpolate(feature,size=(h,w),mode='bilinear',align_corners=True)
            key_q = eval('self.key_q'+str(idx))(feature).view(bs,32,-1) 
            val_q = eval('self.value_q'+str(idx))(feature)   
            for i in range(bs):
                kq = key_q[i].unsqueeze(0).permute(0,2,1)   
                vq = val_q[i].unsqueeze(0)                  
       
                p = torch.matmul(kq,key_t.view(ncls,32,-1))   

                if self.sigmoid_attn:
                    if self.training:
                        storage = get_event_storage()
                        curr_prog = (storage.max_iter - storage.iter * 2) / storage.max_iter
                        temperature = max(self.temperature * curr_prog, 2)
                    else:
                        temperature = 2
                    p = (self.attn_bnn(idx)(p) / temperature).sigmoid()
                else:
                    p = F.softmax(p,dim=1)
                
                full_attention[i].append(F.interpolate(p.permute(0,2,1).view(ncls, att_h * att_w, att_h, att_w), size=(H,W), mode='bilinear', align_corners=True))

                val_t_out = torch.matmul(val_t.view(ncls,128,-1),p).view(ncls,128,att_h,att_w)  
                for j in range(ncls):
                    if(j==0):
                        final_2 = torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)
                    else:
                        final_2 += torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)
                if(i==0):
                    final_1 = final_2
                else:
                    final_1 = torch.cat((final_1,final_2),dim=0)
            final_1 = F.interpolate(final_1,size=(H,W),mode='bilinear',align_corners=True)
            if self.sum:
                final_1 = eval('self.bnn'+str(idx))(final_1)
          
            output.append(final_1)
        

        if self.sum:
            for i in range(len(output)):
                output[i] = self.combine(torch.cat((features[i],output[i]),dim=1))
        output = tuple(output)
        
        return output, full_attention