import os
import glob


def main():
    # Images are divided into vehicles and non-vehicles folders with subfolders
    # Read in image file names for all vehicle data
    basedir = 'data/vehicles/'
    # Different folders represent different sources for images e.g. GTI, KITTI
    image_dirs = os.listdir(basedir)
    cars = []
    for img_dir in image_dirs:
        cars.extend(glob.glob(basedir+img_dir+'/*'))

    print('Number of vehicle images found:', len(cars))
    # Save all vehicle file names to cars.txt
    with open('cars.txt', 'w') as f:
        for fname in cars:
            f.write(fname+'\n')

    # Do the same for non-vehicle images
    basedir = 'data/non-vehicles/'
    image_dirs = os.listdir(basedir)
    non_cars = []
    for img_dir in image_dirs:
        non_cars.extend(glob.glob(basedir+img_dir+'/*'))

    print('Number of non-vehicle images found:', len(non_cars))
    # Save all non-vehicle file names to non_cars.txt
    with open('non_cars.txt', 'w') as f:
        for fname in non_cars:
            f.write(fname+'\n')


if __name__=='__main__':
    main()