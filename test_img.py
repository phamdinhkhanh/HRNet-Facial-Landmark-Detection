import cv2
from utils_inference import get_lmks_by_img, get_model_by_name, get_preds, decode_preds, crop
from utils_landmarks import show_landmarks, get_five_landmarks_from_net, alignment_orig

model = get_model_by_name('WFLW', device='cuda')

img = cv2.imread('khanh6.jpg')
lmks = get_lmks_by_img(model, img) 
show_landmarks(img, lmks)