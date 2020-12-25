# Hunched-Detection-with-Jetson-Nano
Hunched Detection with Jetson Nano using deep neural networks

Sitting on a chair for studying, reading, working with a laptop, etc. are the habits of human of century 21.
It can cause serious injuries for a long time because the back will tend to a hunched position. In this repository, I've provided the package using deep neural networks that can use in Jetson Nano (also on any computer) to avoid sitting in the hunched position.
For using this package, It is necessary to have a webcam besidw the chair. The network had been learnt with two classes: "Not Hunched" and "Hunched". when the position of the back is being tended to the hunched position, the position of the back will be detected with the networks and it can possible to alarm to a user to avoid setting in a hunched position. Finally, by using this package, we can avoid many of the injuries that cause sitting in a hunched position.


# Requirements
Webcam
Jetson Nano (or any other computer)
Python, PyTorch, OpenCV, NumPy, Pillow, 
