The project implements a deep learning-based Convolutional Neural Network (**CNN**) model and a transfer learning-based **VGG16** model to **predict steering angle** in **automated vehicles**. **Microsoft AirSim** has been used for data acquisition and evaluation of the model. During the test run, the CNN and VGG16 model achieved an autonomy of 92% and 94% respectively. Performance is compared with the **NVIDIA PilotNet** baseline, showing **improved autonomy and efficiency with less data**. 

# Table of Contents
- [Introduction](https://github.com/FariaParvinMegha/Steering_angle_prediction/blob/master/README.md#introduction)
- [Features](https://github.com/FariaParvinMegha/Steering_angle_prediction/blob/master/README.md#features)
- [Architecture](https://github.com/FariaParvinMegha/Steering_angle_prediction/blob/master/README.md#architecture)
- [Dataset and Preprocessing](https://github.com/FariaParvinMegha/Steering_angle_prediction/blob/master/README.md#dataset-and-preprocessing)
- [Model Training and Testing](https://github.com/FariaParvinMegha/Steering_angle_prediction/blob/master/README.md#model-training-and-testing)

# Introduction
The project addresses the limitations of modular autonomous vehicle systems by adopting an end-to-end learning approach where perception, prediction, and planning are jointly optimized. The proposed CNN and VGG16 models predict steering angles directly from camera input, reducing computational complexity and cumulative module errors.

# Features
- End-to-End Prediction: Takes raw images as input and outputs steering angles.
- Data Simulation: Uses Microsoft AirSim for realistic vehicle environments.
- Improved Autonomy: Achieves 92% (CNN) and 94% (VGG16) test autonomy.
- Error Stabilization: Uses moving average of frames to minimize vehicle shaking.

# Architecture
- CNN Model: Custom deep convolutional model for feature extraction.
- VGG16 Model: Transfer learning-based model fine-tuned for steering control.
- Loss Function: Mean Squared Error (MSE).
- Evaluation Metric: Autonomy — percentage of time vehicle drives without intervention.

# Dataset and Preprocessing
- Images Collected: 200,000; 96,000 selected post-filtering.
- Preprocessing: Cropping (66×200 px), resizing, balancing, and augmentation (brightness, shadow, shift, flip).

# Model Training and Testing
Both models were trained on normalized steering angles and evaluated within the simulator. FPS and autonomy metrics were compared against PilotNet.

  | Model    | Autonomy (With Obstacles) | Autonomy (Without Obstacles) | FPS |
| -------- | ------------------------- | ---------------------------- | --- |
| PilotNet | 86%                       | 92%                          | 15  |
| CNN      | 92%                       | 99%                          | 12  |
| VGG16    | 94%                       | 99%                          | 10  |
