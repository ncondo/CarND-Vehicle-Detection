from skimage.feature import hog
from detect_cars import get_hog_features
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob


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
    
    ystart = 400
    ystop = 656
    scale = 1.5
    # Iterate over test images
    for idx, img_src in enumerate(images):
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale)
        heat_map = apply_threshold(heat_map, 1)
        
        
        plt.imsave('./output_images/heatmap'+str(idx+1)+'.jpg', heat_map)

def display_bboxes(images):
    
    ystart = 400
    ystop = 656
    scale = 1.5
    # Iterate over test images
    for idx, img_src in enumerate(images):
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale)
        heat_map = apply_threshold(heat_map, 1)
        labels = label(heat_map)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        plt.imsave('./output_images/output'+str(idx+1)+'.jpg', draw_img)


if __name__=='__main__':

    test_images = glob.glob('./test_images/test*.jpg')
    display_hog(test_images)



