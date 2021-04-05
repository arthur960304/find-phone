import os
import sys
import cv2
import torch
import numpy as np

from train_phone_finder import Net

WINDOW_SIZE = 44


def sliding_windows(img):
    """
    Get a collection of sliding window of a given image

    : param img: cv2 img
    : return: windows, list of the upper left and lower right coords of window
    """
    h, w = img.shape
    step = 4

    # number of windows in x, y directions
    nx_windows = int((w - 44) / step)
    ny_windows = int((h - 44) / step)

    windows = []
    for i in range(ny_windows):
        for j in range(nx_windows):
            # calculate window position
            start_x = j * step
            end_x = start_x + WINDOW_SIZE
            start_y = i * step
            end_y = start_y + WINDOW_SIZE

            # append window to the list of windows
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


def get_optimal_window(img, windows, model):
    """
    Get the optimal window which has the phone at the center

    : param img: cv2 img
    : param windows: list of potential windows
    : param model: trained PyTorch model
    : return: windows, list of the upper left and lower right coords of window
    """
    max_prob = 0
    optimal_window = []

    for window in windows:
        t_img = img[window[0][1] : window[1][1], window[0][0] : window[1][0]]
        t_img = t_img.reshape(1, 1, 44, 44)
        t_img = t_img / 255.0
        prob = model(t_img)

        # update optimal window
        if prob > max_prob:
            max_prob = prob
            optimal_window = window
    return optimal_window


def predict(img_path, model):
    """
    Predict the phone position given an image

    : param img_path: path to an image
    : return:
        phone_pos_x: float, x-coord of the phone position
        phone_pos_y: float, y-coord of the phone position
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape

    windows = sliding_windows(img_gray)
    optimal_window = get_optimal_window(img_gray, windows, model)

    phone_pos_x = round((float(optimal_window[0][0] + optimal_window[1][0]) / 2) / w, 4)
    phone_pos_y = round((float(optimal_window[0][1] + optimal_window[1][1]) / 2) / h, 4)

    return phone_pos_x, phone_pos_y


def main():
    path = sys.argv[1]

    # load the trained model
    model = Net()
    model.load_state_dict(torch.load("model.pth"))

    # predict phone position
    pos_x, pos_y = predict(path, model)
    print(pos_x, " ", pos_y)


if __name__ == "__main__":
    main()