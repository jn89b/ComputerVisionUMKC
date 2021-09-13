import os
import cv2

import numpy as np
import random
from collections import defaultdict
import pprint
import cProfile # how to use a profiler

import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
    def __init__(self, num_of_classes, num_of_images, num_rand_pixels):
        self._num_of_classes = num_of_classes
        self._num_of_images = num_of_images
        self._num_rand_pixels = num_rand_pixels

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
    

    def show_training_model(self):
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

            """should put this in hsv model"""
            for image in imageinfo_list:
                hsv_val = image.hsv_list
                training_model_list.append(hsv_val)
                training_model_array = np.concatenate([training_model_array, np.array(hsv_val)])
                #print(training_model_array.shape)

            for index, image_key in enumerate(image_file_keys):
                self.image_dict[folder_class_key][image_key] = imageinfo_list[index]

        return training_model_array, self.image_dict


class ImageInfo():
    """
    Gets the information about the image in COLORGB as well as HSV
    self.img_hsv is reshaped to a [m x n, 3] 2d array where m x n is the width and height of the image in pixels
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


class KMeansModel():
    """
    Takes in imagepresenter and its image and formats a training model for HSV color recognition 
    """
    def __init__(self, num_classes, num_images, num_rand_pixels):
        self.k_int = 64

        self.imgpresenter = ImagePresenter(num_classes, num_images, num_rand_pixels)
        self.training_model_array, self.image_dict = self.imgpresenter.show_training_model()

        self.kmeans=KMeans(n_clusters= self.k_int)
        self.kmeans.fit(self.training_model_array)
     
    def find_kmeans_centroid(self):
        labels=self.kmeans.labels_
        #print("lables are: ", labels)
        labels=list(labels)
        #print("size: ", len(labels))
        
        centroid=(self.kmeans.cluster_centers_)/255
        #print(centroid)
        #print("length of centroid:" , len(centroid))
        
        return labels, centroid
        
    def assign_to_cluster(self):
        """assigns pixel to cluster"""
        pass

if __name__ == '__main__':

    #initial inputs
    # initialize inputs    
    num_classes = 15
    num_images = 5
    num_rand_pixels = 255
    k = 64
    
    """initialize """
    hsv_model = KMeansModel(num_classes, num_images, num_rand_pixels)
    #pprint.pprint(hsv_model.image_dict)
    
    labels, centroid = hsv_model.find_kmeans_centroid()
    
    """testing the minimum function"""
    #pr = cProfile.Profile()
    #pr.enable()
    
    image_histogram = {}
    #test_dict = hsv_model.image_dict['cloud']
    test_dict = hsv_model.image_dict
    
    #fig, axes = plt.subplots(num_images, num_images, figsize=(10, 10))
    fig = plt.figure()

    i = 1    
    
    for i in test_dict.keys():
        for key in test_dict[i].keys():
            #print(test_dict[key].img_hsv.shape) 
            X = test_dict[key].img_hsv.reshape(-1,2) #dividing by 2
            Y = centroid.reshape(-1,2)
            #Y is the centroid and X is the trainingdataset track the minimum distance and the index
            #https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
            min_dists, min_dist_idx = scipy.spatial.cKDTree(Y).query(X, 1)
            image_histogram[key] = Y[min_dist_idx].reshape(-1,3) #I need to map this index to the actual histogram value from the centroid value
            
            #plot histograms
            ax = fig.add_subplot(3, 3,  i )
            ax.hist(np.ndarray.flatten(image_histogram[key]), bins = k)
            ax.set_title(key)
            i +=1
    
    plt.tight_layout()
    plt.show()

    #3D HSV plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centroid[:,0], centroid[:,1], centroid[:,2], c='r', marker='o')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    

    
    
