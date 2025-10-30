The project implements a deep learning-based Convolutional Neural Network (**CNN**) model and a transfer learning-based **VGG16** model to **predict steering angle** in **automated vehicles**. **Microsoft AirSim** has been used for data acquisition and evaluation of the model. During the test run, the CNN and VGG16 model achieved an autonomy of 92% and 94% respectively. Performance is compared with the **NVIDIA PilotNet** baseline, showing **improved autonomy and efficiency with less data**. 

# Table of Contents
- [Introduction] (# Introduction)
- [Features] (# Features)
Architecture
Installation
Usage
Model Training
Alert System Integration

# Introduction
The project addresses the limitations of modular autonomous vehicle systems by adopting an end-to-end learning approach where perception, prediction, and planning are jointly optimized. The proposed CNN and VGG16 models predict steering angles directly from camera input, reducing computational complexity and cumulative module errors.
