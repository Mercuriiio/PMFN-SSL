from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from torch.nn.functional import log_softmax
import cv2
import scipy.io as sio
import random

loss = torch.nn.CrossEntropyLoss()

def cal_loss(predict, target):
	n, c, h, w = target.data.shape

	predict = predict.permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
	target = target.permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
	# [262144, 313]
	return loss(predict, torch.max(target, 1)[1])

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if (out_np.ndim == 2):
		out_np = np.tile(out_np[:, :, None], 3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:, :, 0]
	img_a_orig = img_lab_orig[:, :, 1]
	img_b_orig = img_lab_orig[:, :, 2]

	return img_l_orig, img_a_orig, img_b_orig

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def load_np_image(path, is_scale=True):
	im = cv2.imread(path, -1)
	img = cv2.resize(im, (256, 256), interpolation=cv2.INTER_CUBIC)
	if img.ndim == 2:
		img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	if is_scale:
		img = np.array(img).astype(np.float32) / 255.
	return img


def mask_pixel(img, rate):
	masked_img = img.copy()
	mask = np.ones_like(masked_img)
	perm_idx = [i for i in range(np.shape(img)[1] * np.shape(img)[2])]
	random.shuffle(perm_idx)
	for i in range(np.int32(np.shape(img)[1] * np.shape(img)[2] * rate)):
		x, y = np.divmod(perm_idx[i], np.shape(img)[2])
		masked_img[:, x, y, :] = 0
		mask[:, x, y, :] = 0
	masked_img = np.squeeze(np.uint8(np.clip(masked_img, 0, 1) * 255.))
	mask = np.squeeze(np.uint8(np.clip(mask, 0, 1) * 255.))
	img = np.squeeze(np.uint8(np.clip(img, 0, 1) * 255.))
	return masked_img, img, mask
