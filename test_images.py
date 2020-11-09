
import os
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import pandas as pd

from model import CompNet
from utils import create_dir, seeding, make_channel_last
from data import load_data
from crf import apply_crf

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("mask")

    """ Hyperparameters """
    size = (512, 512)
    checkpoint_path = "files/checkpoint.pth"

    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Set path to test dataset
    TEST_DATASET_PATH = "../data/test/EndoTect_2020_Segmentation_Test_Dataset/"
    MASK_PATH = "mask"

    time_taken = []
    for image_name in os.listdir(TEST_DATASET_PATH):

        # Load the test image
        image_path = os.path.join(TEST_DATASET_PATH, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        H, W, _ = image.shape
        ori_image = image
        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        with torch.no_grad():
            # Start time
            start_time = time.time()

            ## Prediction
            pred_y, pred_m = model(image)
            pred_y = torch.sigmoid(pred_y)
            pred_m = torch.sigmoid(pred_m)
            mask = pred_y

            # End timer
            end_time = time.time() - start_time

            time_taken.append(end_time)
            print("{} - {:.10f}".format(image_name, end_time))

            mask = mask[0].cpu().numpy()
            mask = np.squeeze(mask, axis=0)
            mask = mask > 0.5
            mask = mask.astype(np.float32)
            mask = cv2.resize(mask, (W, H))
            mask = apply_crf(ori_image, mask)
            mask = mask * 255.0

            mask_path = os.path.join(MASK_PATH, image_name)
            cv2.imwrite(mask_path, mask)

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)
