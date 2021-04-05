import os
import sys
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WINDOW_SIZE = 44


class Net(nn.Module):
    def __init__(self):
        """
        Initialize the network
        """
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(324, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Define the forward pass of the network

        : param x: ndarray, the input data
        """
        x = torch.FloatTensor(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def crop_img(img, coord, num_samples=50):
    """
    Create a collection of cropped image

    : param img: cv2 img
    : param coord: int, the phone position
    : param num_samples: int, number of samples
    : return:
        phone_imgs: list of cropped phone imgs
        background_imgs: list of cropped background imgs
    """
    h, w = img.shape
    x, y = coord[0], coord[1]
    center_pixel = np.array((int(x * w), int(y * h)))

    # get bounding box and crop img
    upper_left = [center_pixel[0] - WINDOW_SIZE // 2, center_pixel[1] - WINDOW_SIZE // 2]
    lower_right = [center_pixel[0] + WINDOW_SIZE // 2, center_pixel[1] + WINDOW_SIZE // 2]
    phone_img = img[upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]]

    # get samples for phone data
    phone_imgs = []
    for _ in range(num_samples):
        choice = random.choice([1, 2, 3, 4])
        phone_imgs.append(np.rot90(phone_img, choice))

    # get samples for background data
    background_imgs = []
    for _ in range(num_samples):
        background_x = random.randint(0, w - WINDOW_SIZE)
        background_y = random.randint(0, h - WINDOW_SIZE)

        # to avoid overlap with phone img
        while (upper_left[0] - WINDOW_SIZE < background_x < lower_right[0] 
                and upper_left[1] - WINDOW_SIZE < background_y < lower_right[1]):
            background_x = random.randint(0, w - WINDOW_SIZE)
            background_y = random.randint(0, h - WINDOW_SIZE)

        background_img = img[
            background_y : background_y + WINDOW_SIZE,
            background_x : background_x + WINDOW_SIZE,
        ]
        background_imgs.append(background_img)

    return phone_imgs, background_imgs


def get_data(img_dir):
    """
    Prepare training / testing data

    : param img_dir: image directory
    : return:
        X_train: ndarray, training images
        X_test: ndarray, testing images
        y_train: ndarray, training label
        y_test: ndarray, testing label
    """
    # store information to dictionary
    dict_f = {}
    with open(img_dir + "/" + "labels.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            file_name, x, y = line[0], line[1], line[2]
            dict_f[file_name] = np.array([round(float(x), 4), round(float(y), 4)])

    phone_data = []
    background_data = []
    for file_name in os.listdir(img_dir):
        if file_name != "labels.txt":
            img = cv2.imread(img_dir + "/" + file_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            phone_imgs, background_imgs = crop_img(img_gray, dict_f[file_name])
            phone_data.extend(phone_imgs)
            background_data.extend(background_imgs)

    phone_data = np.array(phone_data)
    background_data = np.array(background_data)
    data = np.vstack((phone_data, background_data))
    label = np.hstack((np.ones(len(phone_data)), np.zeros(len(background_data))))

    # shuffle the data
    data, label = shuffle(data, label)

    # split the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    # reshape data to match input format of CNN
    X_train = X_train.reshape(X_train.shape[0], 1, WINDOW_SIZE, WINDOW_SIZE)
    X_test = X_test.reshape(X_test.shape[0], 1, WINDOW_SIZE, WINDOW_SIZE)

    # normalize input data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test


def train(X_train, y_train, num_epochs=50, batch_size=64, lr=0.001, momentum=0.9):
    """
    Train the phone finder model

    : param X_train: ndarray, training data
    : param y_train: ndarray, training label
    : param num_epochs: int, number of epochs
    : param batch_size: int, batch size
    : param lr: float, learning rate
    : param momentum: float, momentum
    : return: model: trained pytorch model
    """
    model = Net()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            # get training data
            inputs = X_train[i : i + batch_size]
            labels = y_train[i : i + batch_size]
            labels = torch.FloatTensor(labels.reshape(labels.shape[0], 1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def main():
    path = sys.argv[1]

    # get data and train the model
    X_train, X_test, y_train, y_test = get_data(path)
    model = train(X_train, y_train)

    # save the model
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()