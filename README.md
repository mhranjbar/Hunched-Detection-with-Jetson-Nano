# Hunched-Detection-with-Jetson-Nano

## Hunched Detection with Jetson Nano using deep neural networks

Sitting on a chair for studying, reading, working with a laptop, etc. are the habits of human of century 21.
It can cause serious injuries for a long time because the back will tend to a hunched position. In this repository, I've provided the package using deep neural networks that can use in Jetson Nano (also on any computer) to avoid sitting in the hunched position.
For using this package, It is necessary to have a webcam besidw the chair. The network had been learnt with two classes: "Not Hunched" and "Hunched". when the position of the back is being tended to the hunched position, the position of the back will be detected with the networks and it can possible to alarm to a user to avoid setting in a hunched position. Finally, by using this package, we can avoid many of the injuries that cause sitting in a hunched position.

![Hunched Detection deep learning](https://raw.githubusercontent.com/mhranjbar/Hunched-Detection-with-Jetson-Nano/main/demo.jpg)

![Hunched Detection deep learning jetson nano](https://raw.githubusercontent.com/mhranjbar/Hunched-Detection-with-Jetson-Nano/main/demo2.jpg)

## Technical details

|   | Accuracy | No. of Data |
| ------------- | ------------- | ------------- |
| Train  | 97%  | 163  |
| Test  | 86%  | 31  |

## Requirements
Webcam<br/>
Jetson Nano (or any other computer)<br/>
Python, PyTorch, OpenCV, NumPy, Pillow<br/>


## Usage
- Download the wieghts from google drive: https://drive.google.com/file/d/1QMXbDNjkxYlM_gmMuW6pKXu9tEbeO7Jp/view?usp=sharing

- Run **webcam.py** to run the inference.

- Run **train.py** for training.

- Run **test.py** for testing the images (for example: **python3 test.py --imagePath imagesForTest\test3.jpg**)
