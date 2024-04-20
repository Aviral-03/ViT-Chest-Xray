# Chest X-ray Multi-Classification with CNN, ResNet, and Vision Transformers

This repository contains the code and resources for a comparative study of Convolutional Neural Networks (CNNs), Residual Networks (ResNet), and Vision Transformers for multi-classification in X-ray images.

## Overview

Early and accurate detection of diseases is crucial for improving patient outcomes. This project focuses on utilizing deep learning techniques to classify chest X-ray images into different classes of cancerous cells. We compare the performance of CNNs, ResNet, and Vision Transformers to identify the most effective architecture for this task.

## Dataset

We use the NIH Chest X-ray dataset, which comprises 112,120 X-ray images with disease labels from 30,805 unique patients. The dataset is publicly available and includes labels relevant to cancer diagnosis.

1. Navigate to `data_download.ipynb` and download the data. Ensure that it is downloaded into `input` folder.
2. Delete downloaded zip if needed
3. Change the path in models. `image_dir` `root_dir` and `image_path`, otherwise where applicable.

To run it on smaller dataset. Navigate to:

```
https://www.kaggle.com/datasets/nih-chest-xrays/sample
```
1. Download the files
3. Change the path in models. `image_dir` `root_dir` and `image_path`, otherwise where applicable.


## Installation

To run the code, follow these steps:

1. Clone this repository:

```
git clone https://github.com/your-username/CSC413-Final-Project.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```



## Project Members

- Kaushik Murali
- Isha Surani
- Aviral Bhardwaj
- Ananya Jain

## References

Include any relevant references, papers, or resources here.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
