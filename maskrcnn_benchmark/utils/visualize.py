import cv2
import torch
from maskrcnn_benchmark.utils.events import EventStorage, get_event_storage
import copy
import numpy as np
from torch.nn.functional import interpolate

def visualize_episode(meta_input, meta_info, input, targets, results, writer, coco=False):
        storage = get_event_storage()
        meta_input_cpu = meta_input.cpu() + torch.tensor([102.9801, 115.9465, 122.7717, 0]).view(1, 4, 1, 1)
        input_image = input.tensors.cpu() + torch.tensor([102.9801, 115.9465, 122.7717]).view(1, 3, 1, 1)
        scale_f = lambda x,y : interpolate(x.unsqueeze(0), scale_factor=y).squeeze(0)

        meta_input_image = meta_input_cpu[:,[2,1,0],...] / 256
        input_image = input_image[:,[2,1,0],...]
        meta_input_mask = meta_input_cpu[:,-1,...]

        resized_meta_img = []
        for i, per_cls_info in enumerate(meta_info):
                resized_img = cv2.resize(meta_input_image[i].permute(1,2,0).numpy(), dsize=per_cls_info['img_info'][:2][::-1])
                resized_img = torch.tensor(resized_img).permute(2, 0, 1)
                resized_meta_img.append(resized_img)

                storage.put_image("meta_input/{}".format(i), scale_f(resized_img, 0.4))
        
        for i, (per_trg_gt, per_trg_prop, per_trg_roi, per_trg_prop_mask) in enumerate(zip(targets, results["proposal"], results["roi"], results["prop_logs"]["box_mask"])):
                # detach gt proposal
                per_trg_prop = per_trg_prop[:-1]

                per_trg_prop_mask = per_trg_prop_mask[per_trg_prop.extra_fields['labels'] != 0][...,:3]
                per_trg_prop = per_trg_prop[per_trg_prop.extra_fields['labels'] != 0]
                per_trg_roi = per_trg_roi[per_trg_roi.extra_fields['labels'] != 0]

                prop_idx = torch.randperm(len(per_trg_prop))[:5]
                roi_idx = torch.randperm(len(per_trg_roi))[:5]

                per_trg_prop = per_trg_prop[prop_idx]
                per_trg_roi = per_trg_roi[roi_idx]
                per_trg_prop_mask = per_trg_prop_mask[prop_idx]

                gt_overlay = copy.deepcopy(input_image[i])
                prop_overlay = copy.deepcopy(input_image[i])
                roi_overlay = copy.deepcopy(input_image[i])

                gt_overlay = torch.tensor(overlay_boxes(gt_overlay.numpy().transpose(1,2,0), per_trg_gt.to("cpu"))).permute(2,0,1)
                prop_overlay = torch.tensor(overlay_boxes(prop_overlay.numpy().transpose(1,2,0), per_trg_prop.to("cpu"))).permute(2,0,1)
                roi_overlay = torch.tensor(overlay_boxes(roi_overlay.numpy().transpose(1,2,0), per_trg_roi.to("cpu"))).permute(2,0,1)
                
                storage.put_image("target/{}".format(i), scale_f(torch.cat([gt_overlay, prop_overlay, roi_overlay], dim=-1) / 256, 0.4))
                
                curr_trg_attention = results["attention"][i]
                lvl_attn_ma = torch.zeros(5, device=resized_meta_img[0].device)

                for j, att_per_lvl in enumerate(curr_trg_attention):
                        max_att = att_per_lvl.max(dim=1)[0].flatten(1,-1)
                        for k, per_cls_att in enumerate(max_att):
                                storage.put_histogram("cls{}/lvl{}_att".format(k, j), per_cls_att)
                        storage.put_scalar("att_max/lvl{}".format(j), max_att.mean())
                        lvl_attn_ma[j] = storage.latest_with_smoothing_hint(20)["att_max/lvl{}".format(j)][0]

                for j, prop_mask in enumerate(per_trg_prop_mask):
                        lvl, h, w, = prop_mask.long()
                        curr_prop_att = curr_trg_attention[lvl][...,h,w].reshape(-1, 16, 16)
                        curr_prop_overlay = copy.deepcopy(input_image[i])
                        curr_prop_overlay = torch.tensor(overlay_boxes(curr_prop_overlay.numpy().transpose(1,2,0), per_trg_prop[j:j+1].to("cpu"))).permute(2,0,1)

                        storage.put_image("proposal{}/proposal".format(j), scale_f(curr_prop_overlay / 256, 0.4))

                        for k, (meta_img, per_cls_info) in enumerate(zip(resized_meta_img, meta_info)):
                                meta_att = interpolate(curr_prop_att[k].view(1,1,16,16), per_cls_info['img_info'][:2]).squeeze(0).cpu()

                                storage.put_image("proposal{}/attention/{}".format(j,k), scale_f(meta_img * meta_att, 0.4))

                break

        writer.write(storage)


def compute_colors_for_labels(labels):
        """
        Simple function that adds fixed colors depending on the class
        """

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = (labels[:, None] + 10) * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

def overlay_boxes(image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
                image (np.ndarray): an image as returned by OpenCV
                predictions (BoxList): the result of the computation by the model.
                        It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        predictions = predictions.convert("xyxy")
        boxes = predictions.bbox

        colors = compute_colors_for_labels(labels).tolist()

        image = np.ascontiguousarray(image, dtype='uint8')
        for box, color in zip(boxes, colors):
                box = box.to(torch.int64)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                test = cv2.rectangle(
                        image, tuple(top_left), tuple(bottom_right), tuple(color), 3
                )
        image = image.astype('float32')

        return image
