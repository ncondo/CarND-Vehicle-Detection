from skimage.feature import hog
from features import get_hog_features
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import pickle
import numpy as np
import cv2
import glob
from detect_cars import find_cars, apply_threshold, draw_labeled_bboxes


def display_hog(images):

    # Define feature parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    for idx, img_src in enumerate(images):
        img = mpimg.imread(img_src)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        _, hog_image = get_hog_features(img[:,:,0], orient, pix_per_cell,
                                cell_per_block, vis=True, feature_vec=False)
        plt.imsave('./output_images/hog'+str(idx+1)+'.jpg', hog_image)


def display_heatmap(images):

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    hist_bins = 32
    
    ystart = 400
    ystop = 656
    scale = 1.5

    # Iterate over test images
    for idx, img_src in enumerate(images):
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale, ystart, ystop, pix_per_cell,
                                cell_per_block, orient, spatial_size, hist_bins)
        heat_map = apply_threshold(heat_map, 1)
        
        
        plt.imsave('./output_images/heatmap'+str(idx+1)+'.jpg', heat_map)

def display_bboxes(images):

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    hist_bins = 32
    
    ystart = 400
    ystop = 656
    scale = 1.5
    # Iterate over test images
    for idx, img_src in enumerate(images):
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale, ystart, ystop, pix_per_cell,
                                cell_per_block, orient, spatial_size, hist_bins)
        heat_map = apply_threshold(heat_map, 1)
        labels = label(heat_map)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        plt.imsave('./output_images/output'+str(idx+1)+'.jpg', draw_img)


if __name__=='__main__':

    # Load list of images to show viz
    test_images = glob.glob('./test_images/test*.jpg')
    # Display hog features on test images
    display_hog(test_images)
    # Display heatmap
    display_heatmap(test_images)
    # Display bboxes
    display_bboxes(test_images)

