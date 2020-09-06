import argparse
import time
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms

import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.mtcnn import detect_faces, show_bboxes
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset, get_preprocess
from lib.core import function
from utils_inference import get_lmks_by_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

def main():
    # Step 1: load model
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    # Step 2: detect face and predict landmark
    transform = transforms.Compose([transforms.ToTensor()])
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break

        height, width = img.shape[:2]

        bounding_boxes, landmarks = detect_faces(img)
        dataset = get_preprocess(config)
        # print('--------bboxes: ', bounding_boxes)
        for box in bounding_boxes:
            # x1, y1, x2, y2, _ = list(map(int, box))
            score = box[4]
            x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h])*1.3)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = img[y1:y2, x1:x2]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            
            # center_w = (x1+x2)/2
            # center_h = (y1+y2)/2
            # center = torch.Tensor([center_w, center_h])
            # input = img[y1:y2, x1:x2, :]
            # input = dataset._preprocessing(dataset, img=img, center=center, scale=1.0)
            landmarks = get_lmks_by_img(model, cropped) 
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            for (x, y) in landmarks.astype(np.int32):
                cv2.circle(img, (x1+x, y1+y), 2, (255, 255, 255))

        cv2.imshow('0', img)
        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    # python3 tools/camera.py --cfg experiments/aflw/face_alignment_aflw_hrnet_w18.yaml --model-file hrnetv2_pretrained/HR18-AFLW.pth
    # python3 tools/camera.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file hrnetv2_pretrained/HR18-WFLW.pth
    main()