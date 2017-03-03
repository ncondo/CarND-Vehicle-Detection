# Vehicle Detection for Self-Driving Cars

The goal of this project is to produce a robust pipeline for detecting vehicles from a front-facing camera on a car. The pipeline should output a visual display of bounding boxes around each detected car.

![Original Image](test_images/test_example1.jpeg)   ![Output Image](output_images/output_example1.jpeg)


## Files and Usage

1. combine_data.py
    * Contains code for combining all image file names from the different data sources to make extracting data during training easy.
    * `python combine_data.py` will save the combined data in data/cars.txt and data/non_cars.txt.
2. features.py
    * Contains code for extracting the features on which to train a classifier for detecting cars.
    * `python features.py` will save the features in separate files named car_features.p and noncar_features.p
3. classifier.py
    * Contains code for training a linear SVM classifier on the extracted car and non-car features.
    * `python classifier.py` will save the classifier and scaler to use when making predictions.
4. visualize.py
    * Contains code for visualizing different features, heatmaps, and bounding boxes from test images.
    * `python visualize.py` will save images as described above in the output_images folder.
5. detect_cars.py
    * Contains code for detecting vehicles using the trained classifier and drawing bounding boxes around them in a video.
    * `python detect_cars.py` will run the classifier on project_video.mp4 and save a new video named project_video_output.mp4 with bounding boxes drawn around detect cars.

## Solution

### Overview

The steps taken to complete this project are as follows:

* Perform feature extraction using Histogram of Oriented Gradients (HOG), apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Train a Linear SVM classifier on the extracted features.
* Implement a sliding-window technique with the trained classifier to detect vehicles in an image.
* Create a heatmap of recurring detections to reject outliers.
* Output visual display of bounding boxes around detected vehicles in a video stream.


### Feature Extraction