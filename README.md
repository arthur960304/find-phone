# Find Phone

The goal of this visual object detection system is to find a location of a phone dropped on the floor from a single RGB camera image.

<p align="center">
  <img width="490" height="326" src="https://github.com/arthur960304/find-phone/blob/main/images/97.jpg">
</p>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Built With

* Python 3.6.10

* PyTorch 1.7.0

* OpenCV 4.4.0

* numpy >= 1.16.2

* matplotlib >= 3.1.1

## Code Organization
```
.
├── train_phone_finder.py   # Train the phone detector
├── find_phone.py           # Predict the phone location
├── description.pdf         # File providing implementation details
├── images                  # Training data
└── README.md
```

## Tests

The train_phone_finder.py file receives the path to training data directory to train the model

```
python train_phone_finder.py imgs
```

The find_phone.py file receives the path to the target image, and it will print the normalized coordinates of the phone detected on this test image

```
python train_phone_finder.py imgs/51.jpg
0.2551 0.6013
```

## Results

<p align="center">
  <img width="600" height="400" src="https://github.com/arthur960304/find-phone/blob/main/result.png"><br/>
  <em>Predicted bounding box of the phone.</em>
</p>

## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
