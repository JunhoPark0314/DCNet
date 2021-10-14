from collections import defaultdict
from maskrcnn_benchmark.layers.batch_norm import SharedBatchNorm2d
import torch
import torch.nn.functional as F
import torch.nn as nn
from maskrcnn_benchmark.utils.events import get_event_storage
 
class DenseRelationDistill(nn.Module):

    def __init__(self, indim, keydim, valdim, dense_sum=False, sigmoid_attn=False, no_padding=False, per_level_bn=False, normalize_concat=False, add_one_more=True):
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
        self.normalize_concat = normalize_concat
        self.add_one_more = add_one_more
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
        
        self.temperature = 10
        if self.sigmoid_attn:
            self.sigmoid_attn_bnn = nn.ModuleList([
                nn.BatchNorm1d(bn_dim) for _ in range(5)])

            if per_level_bn:
                self.attn_bnn = lambda x: self.sigmoid_attn_bnn[x]
            else:
                self.attn_bnn = lambda x: self.sigmoid_attn_bnn[0]
        
        if self.normalize_concat:
            """
            self.concat_bnn = nn.ModuleList([
                SharedBatchNorm2d(num_features=bn_dim, num_shared=2) for _ in range(5)
            ])
            """
            self.key_bnn = SharedBatchNorm2d(num_features=keydim, num_shared=2)
            self.val_bnn = SharedBatchNorm2d(num_features=valdim, num_shared=2)
            self.level_bnn = nn.BatchNorm2d(num_features=bn_dim)
   
    def forward(self,features,attentions):
        """
        old_features = features
        old_attentions = (attentions)
        if self.normalize_concat and isinstance(attentions, dict) is False:
            features = list([self.concat_bnn(f, share_id=0) for i, f in enumerate(features)])
            attentions = self.attn_bnn(attentions.detach(), share_id=0)
        else:
            features = list(features)
        """
        storage = get_event_storage()
        if storage.iter % 10 == 0:
            for i,f in enumerate(features):
                print('f{}'.format(i), torch.std_mean(f))
            print('a',torch.std_mean(attentions))

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
        if self.normalize_concat:
            key_t = self.key_bnn(key_t, share_id=1)

        att_h, att_w = val_t.shape[2:]
        for idx in range(len(features)):
            feature = features[idx]
            bs = feature.shape[0]       
            H , W = feature.shape[2:]
            feature = F.interpolate(feature,size=(h,w),mode='bilinear',align_corners=True)
            key_q = eval('self.key_q'+str(idx))(feature) 
            val_q = eval('self.value_q'+str(idx))(feature)   
            if self.normalize_concat:
                val_q = self.val_bnn(val_q, share_id=0) / len(key_t)
                key_q = self.key_bnn(key_q, share_id=0)
            key_q = key_q.view(bs,32,-1)

            for i in range(bs):
                kq = key_q[i].unsqueeze(0).permute(0,2,1)   
                vq = val_q[i].unsqueeze(0)                  
       
                p = torch.matmul(kq,key_t.view(ncls,32,-1))   

                if self.training:
                    storage = get_event_storage()
                    curr_prog = (storage.max_iter - storage.iter * 2) / storage.max_iter
                    temperature = max(self.temperature * curr_prog, 2)
                else:
                    temperature = 2
                if self.sigmoid_attn:
                    p = (self.attn_bnn(idx)(p) / temperature).sigmoid()
                else:
                    p = F.softmax(p / temperature,dim=1)
                
                full_attention[i].append([F.interpolate(p.view(ncls, att_h * att_w, att_h, att_w), size=(H,W), mode='bilinear', align_corners=True), (att_h, att_w)])

                val_t_out = torch.matmul(val_t.view(ncls,128,-1),p).view(ncls,128,att_h,att_w)  

                if self.normalize_concat:
                    val_t_out = self.val_bnn(val_t_out, share_id=1)

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
        
        if self.add_one_more:
            if self.sum:
                for i in range(len(output)):
                    if self.normalize_concat:
                        features[i] = self.level_bnn(features[i])
                    output[i] = self.combine(torch.cat((features[i],output[i]),dim=1))
        output = tuple(output)
        
        return output, full_attention