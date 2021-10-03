import cv2
import torch
from maskrcnn_benchmark.utils.events import EventStorage, get_event_storage
import copy
import numpy as np
from torch.nn.functional import interpolate
scale_f = lambda x,y : interpolate(x.unsqueeze(0), scale_factor=y).squeeze(0)

def visualize_episode(meta_input, meta_info, input, targets, results, writer, coco=False):
	storage = get_event_storage()
	meta_input_cpu = meta_input.cpu() + torch.tensor([102.9801, 115.9465, 122.7717, 0]).view(1, 4, 1, 1)
	input_image = input.tensors.cpu() + torch.tensor([102.9801, 115.9465, 122.7717]).view(1, 3, 1, 1)

	meta_input_image = meta_input_cpu[:,[2,1,0],...] / 256
	input_image = input_image[:,[2,1,0],...]
	meta_input_mask = meta_input_cpu[:,-1,...]

	resized_meta_img = []
	for i, per_cls_info in enumerate(meta_info):
		resized_img = cv2.resize(meta_input_image[i].permute(1,2,0).numpy(), dsize=per_cls_info['img_info'][:2][::-1])
		resized_img = torch.tensor(resized_img).permute(2, 0, 1)
		resized_meta_img.append(resized_img)

		storage.put_image("meta_input/{}".format(i), scale_f(resized_img, 0.4))
	
	for i, (per_trg_gt, per_trg_prop, per_trg_roi, per_trg_prop_mask, curr_trg_attention) in \
		enumerate(zip(targets, results["proposal"], results["roi"], results["prop_logs"]["box_mask"], list(results["attention"].values()))):
		# detach gt proposal
		per_trg_prop = per_trg_prop[:len(per_trg_prop_mask)]
		lvl_attn_ma = torch.zeros(5, device=resized_meta_img[0].device)
		lvl_attn_mstd = torch.zeros(5, device=resized_meta_img[0].device)

		pos_prop_mask = per_trg_prop.extra_fields['labels'] != 0
		pos_roi_mask = per_trg_roi.extra_fields['labels'] != 0

		neg_prop_mask = per_trg_prop.extra_fields['labels'] == 0
		neg_roi_mask = per_trg_roi.extra_fields['labels'] == 0

		pos_trg_prop, pos_trg_roi, pos_trg_prop_mask = sample_result(per_trg_prop, per_trg_roi, per_trg_prop_mask, pos_prop_mask, pos_roi_mask)
		neg_trg_prop, neg_trg_roi, neg_trg_prop_mask = sample_result(per_trg_prop, per_trg_roi, per_trg_prop_mask, neg_prop_mask, neg_roi_mask)

		visualize_detection_result(input_image[i], per_trg_gt, pos_trg_prop, pos_trg_roi, "pos{}".format(i), storage)
		visualize_detection_result(input_image[i], per_trg_gt, neg_trg_prop, neg_trg_roi, "neg{}".format(i), storage)

		attention_histogram(curr_trg_attention, lvl_attn_ma, lvl_attn_mstd, storage)
		
		visualize_attention(pos_trg_prop_mask, curr_trg_attention, input_image[i], pos_trg_prop, storage, lvl_attn_ma, lvl_attn_mstd, resized_meta_img, meta_info, "pos")
		visualize_attention(neg_trg_prop_mask, curr_trg_attention, input_image[i], neg_trg_prop, storage, lvl_attn_ma, lvl_attn_mstd, resized_meta_img, meta_info, "neg")

		break

	writer.write(storage)

def sample_result(per_trg_prop, per_trg_roi, per_trg_prop_mask, prop_mask, roi_mask):
	trg_prop = per_trg_prop[prop_mask]
	trg_roi = per_trg_roi[roi_mask]
	trg_prop_mask = per_trg_prop_mask[prop_mask][...,:3]

	prop_idx = torch.randperm(len(trg_prop))[:5]
	roi_idx = torch.randperm(len(trg_roi))[:5]

	trg_prop = trg_prop[prop_idx]
	trg_roi = trg_roi[roi_idx]
	trg_prop_mask = trg_prop_mask[prop_idx]

	return trg_prop, trg_roi, trg_prop_mask

def attention_histogram(curr_trg_attention, lvl_attn_ma, lvl_attn_mstd, storage):
	for j, att_per_lvl in enumerate(curr_trg_attention):
		max_att = att_per_lvl.max(dim=1)[0].flatten(1,-1)
		for k, per_cls_att in enumerate(max_att):
			storage.put_histogram("lvl{}_att/cls{}".format(j, k), per_cls_att * 256)
		storage.put_scalar("att_max_mean/lvl{}".format(j), max_att.mean())
		storage.put_scalar("att_max_std/lvl{}".format(j), max_att.std())
		lvl_attn_ma[j] = storage.latest_with_smoothing_hint(20)["att_max_mean/lvl{}".format(j)][0]
		lvl_attn_mstd[j] = storage.latest_with_smoothing_hint(20)["att_max_std/lvl{}".format(j)][0]

def visualize_detection_result(input_image, per_trg_gt, per_trg_prop, per_trg_roi, img_idx, storage):
	gt_overlay = copy.deepcopy(input_image)
	prop_overlay = copy.deepcopy(input_image)
	roi_overlay = copy.deepcopy(input_image)

	gt_overlay = torch.tensor(overlay_boxes(gt_overlay.numpy().transpose(1,2,0), per_trg_gt.to("cpu"))).permute(2,0,1)
	prop_overlay = torch.tensor(overlay_boxes(prop_overlay.numpy().transpose(1,2,0), per_trg_prop.to("cpu"))).permute(2,0,1)
	roi_overlay = torch.tensor(overlay_boxes(roi_overlay.numpy().transpose(1,2,0), per_trg_roi.to("cpu"))).permute(2,0,1)
	
	storage.put_image("target/{}".format(img_idx), scale_f(torch.cat([gt_overlay, prop_overlay, roi_overlay], dim=-1) / 256, 0.4))

def visualize_attention(per_trg_prop_mask, curr_trg_attention, input_image, per_trg_prop, storage, lvl_attn_ma, lvl_attn_mstd, resized_meta_img, meta_info, tag):
	meta_att_norm = []
	for j, prop_mask in enumerate(per_trg_prop_mask):
		lvl, h, w, = prop_mask.long()
		curr_prop_att = curr_trg_attention[lvl][...,h,w].reshape(-1, 16, 16)
		curr_prop_overlay = copy.deepcopy(input_image)
		curr_prop_overlay = torch.tensor(overlay_boxes(curr_prop_overlay.numpy().transpose(1,2,0), per_trg_prop[j:j+1].to("cpu"))).permute(2,0,1)

		storage.put_image("{}_proposal_{}/proposal".format(tag,j), scale_f(curr_prop_overlay / 256, 0.4))

		for k, (meta_img, per_cls_info) in enumerate(zip(resized_meta_img, meta_info)):
			meta_att = interpolate(curr_prop_att[k].view(1,1,16,16), per_cls_info['img_info'][:2]).squeeze(0).cpu()
			meta_att = (((meta_att - lvl_attn_ma[j]) / lvl_attn_mstd[j]) * 5).sigmoid()

			storage.put_image("{}_proposal_{}/attention/{}".format(tag,j,k), scale_f(meta_img * meta_att, 0.4))
			meta_att_norm.append((meta_att ** 2).mean().sqrt())

	storage.put_histogram("{}_att_norm".format(tag), torch.stack(meta_att_norm))

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
