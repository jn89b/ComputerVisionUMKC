import os
from PIL import Image
import cv2
import glob

import numpy as np
import random
"""
GOALS and resources
https://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
https://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/
https://realpython.com/k-means-clustering-python/

https://sce.umkc.edu/faculty-sites/lizhu/teaching/2021.fall.vision/notes/tut01.pdf
https://sce.umkc.edu/faculty-sites/lizhu/teaching/2021.fall.vision/notes/lec03.pdf

group numbers into k clustesr
pick k points randomly in the data/image
set these k points as cluster

loop through every data point in data 
calculate distance to each cluster
find minimum cluster and assign to datapoint to that cluster

calculate new center of k clusters by finding new average from distances of points assigned to cluster
stop when k cluster value converges to desired value

plot cluster points

"""

class ImagePresenter():
    """
    image presenter goes to file dir where images are and picks random n unique random images 
    in the folder directory
    """
    def __init__(self, num_of_classes, num_of_images):
        self._num_of_classes = num_of_classes
        self._num_of_images = num_of_images
    
        self.cwd = os.getcwd()
        self.image_path = os.path.join(self.cwd, './NWPU_first_fifteen')

        self.image_dict = {}

    def get_folder_class_names(self):
        """method folder_class_names returns names of classes and appends to list"""
        dirs = os.listdir(self.image_path)
        folder_class_list = []
        for folder_name in dirs:
            folder_class_list.append(folder_name)

        return folder_class_list    

    def open_folder_class(self, class_name):
        print(self.image_path)
        test_name = './' + class_name
        folder_class_path = os.path.join(self.image_path, test_name)
   
        return folder_class_path
    
    def append_class_to_dict(self, class_list):
        """method append_class_to_dict recieves class name list and appends to dict"""
        self.image_dict = {class_name: [] for class_name in class_list}
        
        return self.image_dict

    def pick_random_images(self, folder_class_path):
        """pick random images from class inputs are the folder directory to open up to"""
        os.chdir(folder_class_path)
        random_images_list = random.sample(os.listdir(folder_class_path), self._num_of_images) # 45 is number of images
        
        return random_images_list
        
    def load_image(self):
        pass

    def append_image(self):
        pass

    def main(self):
        """main method class"""
        class_list = self.get_folder_class_names()
        self.append_class_to_dict(class_list)
        
        # for each key in dictionary loop through and append image name lists to dictionary
        for key in self.image_dict:
            folder_class_dir = self.open_folder_class(key)
            random_images_list = self.pick_random_images(folder_class_dir)
            self.image_dict[key].append(random_images_list)
        
        return self.image_dict
        
        
class ImageInfo():
    """
    get image
    pick 256 random points from image
    get its rgb info
    convertg rgb to hsv 
    return hsv values
    """
    def __init__(self,img_path):
        self.img = cv2.imread(img_path)
        self.image_shape = self.img.shape #h,w,size        
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        
class Kmeans():

    def __init__(self):
        self.k_int = 64
        self.imgpresenter = ImagePresenter()
        
        self.train_model = []     
        
    def calculate_euc_dist(self):
        pass


if __name__ == '__main__':
    num_classes = 15
    num_images = 10
    imgpresent = ImagePresenter(num_classes, num_images)
    imgpresent.main()
    
    
    
    path = r"C:\Users\jn89b\Box\School\ComputerVision\hw-1\NWPU_first_fifteen\airplane\airplane_001.jpg"
    imginfo = ImageInfo(path)

    #image test

