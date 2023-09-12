# Task 3: ResoluteAI - Object Detection and Counting
This repository contains the code for the task 3 of the ResoluteAI screening process.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Model Weights](#model-weights)
- [Installation and Usage](#installation-and-usage)

## Problem Statement
The problem statement is to detect and count the number of objects in an image. The images consist of dense environment like inside of a freezer in a super market.

## Approach
I trained a custom YOLOv5x with SKU-110K dataset for educational and research purposes a few months back. I used the same model to detect objects in dense environments. For more information you check out this repository.

Link: [Dense-Object-Detection](https://github.com/suryanshgupta9933/Dense-Object-Detection)

## Model Weights
The weights for the model are not added in the repository due to their large size but can be downloaded from [here](https://drive.google.com/file/d/1BRlXZD9MqYAYYnciMRQ50Mht9kBncf0l/view?usp=sharing).

> Note: You need to add the weights in the root directory of the repository.

## Installation and Usage
1. Clone the repository
```bash
git clone https://github.com/suryanshgupta9933/task3-resoluteai.git
```
2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Run the script
```bash
streamlit run app.py
```

## Model Performance
The model performs really well on the images with dense environments. Detections on the sample data is shown below:

Index | Sample Data             |  Resultant Image
:----:|:-----------------------:|------------------------
1.|![freezer_1](sample_images/freezer_image%20(1).jpg)|![result_1](assets/result_1.jpg)
2.|![freezer_2](sample_images/freezer_image%20(2).jpg)|![result_2](assets/result_2.jpg)

## Results
The default settings for confidence threshold and IoU threshold are 0.25 and 0.1 respectively. This can be changed in the sidebar of the web app.

> Note: The increasing the confidence threshold will result in lesser number of detections and increasing the IoU threshold will result in more overlapping bounding boxes leading to incorrect counting of objects.

The results for the sample data is shown below:
