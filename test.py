
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score

from model import CompNet
from utils import create_dir, seeding, make_channel_last
from data import load_data

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jaccard_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_recall = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_precision = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary', zero_division=0)

    return [score_jaccard, score_f1, score_recall, score_precision]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    path = "/media/nikhil/ML/ml_dataset/Kvasir-SEG/"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    # """ CVC-ClinicDB """
    # test_x = sorted(glob("/media/nikhil/ML/ml_dataset/CVC-612/images/*"))
    # test_y = sorted(glob("/media/nikhil/ML/ml_dataset/CVC-612/masks/*"))

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

    """ Testing """
    metrics_score = [0.0, 0.0, 0.0, 0.0]

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        ori_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        ## Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        ori_mask = mask
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        ## Gray
        gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, size)
        ori_gray = gray
        # gray = np.expand_dims(gray, axis=0)
        # gray = gray/255.0
        # gray = np.expand_dims(gray, axis=0)
        # gray = gray.astype(np.float32)
        # gray = torch.from_numpy(gray)
        # gray = gray.to(device)

        with torch.no_grad():
            pred_y, pred_m = model(image)
            pred_y = torch.sigmoid(pred_y)
            pred_m = torch.sigmoid(pred_m)

            score = calculate_metrics(mask, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            ## Mask
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = pred_y * 255
            # pred_y = np.transpose(pred_y, (1, 0))
            pred_y = np.array(pred_y, dtype=np.uint8)

            ## Gray
            pred_m = pred_m[0].cpu().numpy()
            pred_m = np.squeeze(pred_m, axis=0)
            pred_m = pred_m * 255
            # pred_m = np.transpose(pred_m, (1, 0))
            pred_m = np.array(pred_m, dtype=np.uint8)

        ori_img     = ori_img
        ori_mask    = mask_parse(ori_mask)
        pred_y      = mask_parse(pred_y)
        ori_gray    = mask_parse(ori_gray)
        pred_m      = mask_parse(pred_m)
        sep_line    = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img, sep_line,
            ori_mask, sep_line,
            pred_y, sep_line,
            ori_gray, sep_line,
            pred_m
        ]

        cat_images = np.concatenate(tmp, axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f}")
