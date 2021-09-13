import os
from PIL import Image
import glob

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
    image presenter goes to file dir where images are, gets num_classes and num of images in each class
    """
    def __init__(self, num_of_classes, num_of_images):
        self._num_of_classes = num_of_classes
        self._num_of_images = num_of_images
    
        self.cwd = os.getcwd()
        self.img_path = os.path.join(self.cwd, './NWPU_first_fifteen')

        self.image_dict = {}

    def get_folder_class_names(self):
        """method folder_class_names returns names of classes and appends to list"""
        dirs = os.listdir(self.img_path)
        folder_class_list = []
        for folder_name in dirs:
            folder_class_list.append(folder_name)


        return folder_class_list    

    def append_class_to_dict(self, class_list):
        """method append_class_to_dict recieves class name list and appends to dict"""
        self.image_dict = {class_name: None for class_name in class_list}
        return self.image_dict

    def get_image_os(self):
        """pick random image from class"""
        pass
        
    def load_image(self):
        pass

    def append_image(self):
        pass

    def main(self):
        """main method class"""
        class_list = self.get_folder_class_names()
        self.image_dict = self.append_class_to_dict(class_list)

class ImageInfo():
    """
    get image
    pick 256 random points from image
    get its rgb info
    convertg rgb to hsv 
    return hsv values
    """
    def __init__(self,img_path):
        self.img = Image.open(img_path)

        self.h, self.w = self.img.size
        print("open image")
    
    def get_random_points(self):
        for i in range(len(self.h)):
            #convert rgb to hsv
            #pick random point of hsv and store
            pass

class Kmeans():
    def __init__(self):
        self.k = 64

    def calculate_euc_dist(self):
        pass


#Kmeans clustering
def group_numbers():
    pass

def pick_points():
    pass

def set_points():
    pass



def find_min_dist():
    pass


if __name__ == '__main__':
    num_classes = 15
    num_images = 40
    imgpresent = ImagePresenter(num_classes, num_images)
    imgpresent.main()
    
    path = r"C:\Users\jn89b\Box\School\ComputerVision\hw-1\NWPU_first_fifteen\airplane\airplane_001.jpg"
    imginfo = ImageInfo(path)

    #image test

