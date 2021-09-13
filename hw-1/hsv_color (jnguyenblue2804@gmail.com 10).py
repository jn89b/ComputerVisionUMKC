import os
import cv2
import glob

import numpy as np
import random
from collections import defaultdict
import pprint
import cProfile # how to use a profiler

from sklearn.cluster import KMeans

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
    and stores the information of the folder class with the random images in self.image_dict
    """
    def __init__(self, num_of_classes, num_of_images):
        self._num_of_classes = num_of_classes
        self._num_of_images = num_of_images
        self._num_rand_pixels = int(255)

        self.cwd = os.getcwd()
        self.image_path = os.path.join(self.cwd, './NWPU_first_fifteen')

        self.image_dict = defaultdict(lambda: defaultdict(dict))

    def get_folder_class_names(self):
        """method folder_class_names returns names of classes and appends to list"""
        dirs = os.listdir(self.image_path)
        folder_class_list = []
        for folder_name in dirs:
            folder_class_list.append(folder_name)

        return folder_class_list    

    def open_folder_class(self, class_name):
        folder_name = './' + class_name
        folder_class_path = os.path.join(self.image_path, folder_name)
   
        return folder_class_path
    
    def pick_random_images(self, folder_class_path):
        """pick n number of random images in class folder directory and return as a list"""
        os.chdir(folder_class_path)
        random_images_list = random.sample(os.listdir(folder_class_path), self._num_of_images)         
        
        return random_images_list
    

    def main(self):
        """main method class"""
        class_list = self.get_folder_class_names()
        self.image_dict = {class_key: [] for class_key in class_list} # set as key
        # for each key in dictionary loop through and append image name lists to dictionary
        training_model_list = []
        training_model_array = np.empty((0, 3))
        
        for folder_class_key in self.image_dict:
            folder_class_dir = self.open_folder_class(folder_class_key)
            random_images_list = self.pick_random_images(folder_class_dir)
            image_file_keys = {image_key: [] for image_key in random_images_list}
            self.image_dict[folder_class_key] = image_file_keys
            
            """this line of code takes forever to read the objects"""
            images = [cv2.imread(os.path.join(folder_class_dir, random_image)) for random_image in random_images_list]
            imageinfo_list = [ImageInfo(image, self._num_rand_pixels) for image in images] 
            #training_model = [imageinfo.hsv_list for imageinfo in imageinfo_list] 
            for image in imageinfo_list:
                hsv_val = image.hsv_list
                training_model_list.append(hsv_val)
                training_model_array = np.concatenate([training_model_array, np.array(hsv_val)])
                print(training_model_array.shape)

            for index, image_key in enumerate(image_file_keys):
                self.image_dict[folder_class_key][image_key] = imageinfo_list[index]

        return self.image_dict, training_model_list, training_model_array


class ImageInfo():
    """
    Gets the information about the image in COLORGB as well as HSV
    has methods to return random coordinates with a permuation based on n_random_inputs
    has a method to a list of the HSV values of the random coordinates
    """
    def __init__(self,img ,n_random_points):
        
        self.n_random_points = n_random_points
        self.img = img
        
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.img_hsv = self.img_hsv.reshape((-1,3)) #reshape image from 
        self.img_hsv = np.float32(self.img_hsv) #convert to float 32   

        self.random_coordinates = self.get_random_coordinates()
        self.hsv_list = self.get_hsv_list(self.random_coordinates)

    def get_random_coordinates(self):
        """pick random permutated x,y coordinates of pixels and returns in 2d array"""
        """
        random_x = np.random.permutation(self.n_random_points)
        random_y = np.random.permutation(self.n_random_points)
        random_coordinates = np.column_stack((random_x, random_y))
        """
        
        random_coordinates = np.random.permutation(self.n_random_points)
        #pprint.pprint(random_coordinates)
        return random_coordinates

    def get_hsv_list(self, random_coordinates):
        """gets h,s,v from coordinates specified"""
        hsv_val_list = []
        
        #hsv_array = np.empty((random_coordinates, random_coordinates, 3)) # 3d array
        for i in range(len(random_coordinates)):
            
            hsv_val = self.img_hsv[random_coordinates[i],:]
            hsv_val_list.append(hsv_val)

        return hsv_val_list


class HSVModel():
    """
    Takes in imagepresenter and its image and formats a training model for HSV color recognition 
    """
    def __init__(self):
        self.k_int = 64
        self.imgpresenter = ImagePresenter()

        self.train_model = []     
        
    def calculate_euc_dist(self):
        pass

        
if __name__ == '__main__':

    #pr = cProfile.Profile()
    #pr.enable()    
    num_classes = 15
    num_images = 2
    imgpresent = ImagePresenter(num_classes, num_images)
    test_dict, test_list, training_model_array = imgpresent.main()      
    print("size of array is:" , training_model_array.shape)

    #kmeans to return index and color table]
    # define stopping criteria
    K = 64
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret,label,center=cv2.kmeans(np.float32(training_model_array),K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    """

    kmeans=KMeans(n_clusters=64)
    s=kmeans.fit(training_model_array)
    
    labels=kmeans.labels_
    print("lables are: ", labels)
    labels=list(labels)
    print("size: ", len(labels))
    
    centroid=kmeans.cluster_centers_
    print(centroid)
    print("length of centroid:" , len(centroid))

    #pr.print_stats(sort='time')
