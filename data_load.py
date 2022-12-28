import glob
import os
import torch
import random
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torchvision import transforms


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, extend=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            extend (int, optional): To extend dataset
        """
        self.key_pts_frame = pd.read_csv(csv_file)

        if(extend>1):
            '''
            Used to make multiple copies of the same dataset, whenever the pre_processing
            is defined (in method __getitem__),images will ranfomly be rotated/flipped(x,y). This is done to extend 
            the dataset
            '''
            for i in range(1,extend):
                df_copy =  self.key_pts_frame
                self.key_pts_frame = pd.concat([self.key_pts_frame, df_copy])
                
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)
       

        pre_processed_image, pre_processed_key_points = self.pre_processing(sample['image'], sample['keypoints'])
        
        if type(pre_processed_key_points) == "numpy.ndarray":
            pre_processed_key_points = torch.from_numpy(pre_processed_key_points)
            
        if type(pre_processed_image) == "numpy.ndarray":
            pre_processed_image = torch.from_numpy(pre_processed_image)
        
        sample = {'image': pre_processed_image, 'keypoints': pre_processed_key_points }            

        return sample
    
    def rotate_keypoints(self, keypoints, R, center):
        """Rotate the keypoints using a rotation matrix.

        Parameters
        ----------
        keypoints : numpy.ndarray
            A Nx2 array of keypoints, where N is the number of keypoints
            and each row represents a keypoint with (x, y) coordinates.
        R : numpy.ndarray
            A 2x2 rotation matrix.
        center: numpy.ndarray
            A 2x2 array indicating the origin of rotation/translation

        Returns
        -------
        numpy.ndarray
            A Nx2 array of rotated keypoints.
        """

        # Perform matrix multiplication to rotate the keypoints
        rotated_keypoints = np.dot(keypoints, R.T)

        # Create the translation matrix
        T = np.array([[1, 0, center[0]],
                      [0, 1, center[1]],
                      [0, 0, 1]])


        # Add a third coordinate of 1 to the coordinates
        coordinates = rotated_keypoints[:, :2]
        coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))

        # Perform matrix multiplication to translate the coordinates
        translated_coordinates = np.dot(coordinates, T.T)

        # Return the translated coordinates with the third coordinate removed
        translated_coordinates = translated_coordinates[:, :2]

        # Return the rotated + translated keypoint coordinates with the third coordinate removed
        return translated_coordinates

    def flip_keypoints(self, keypoints, T, center):
        """Flips key points using matrix multiplication.

        Parameters
        ----------
        keypoints : numpy.ndarray
            A Nx2 array of keypoints, where N is the number of keypoints
            and each row represents a keypoint with (x, y) coordinates.


        T: numpy.ndarray
            A 2x2 Translation matrix, [[-1,0],[0,1]] for horizontal flip
            [[1,0],[0,-1]] for vertical flip

        center: numpy.ndarray
            (x,y) coordinates indicating the center of translation

        Returns
        -------
        numpy.ndarray
            A Nx2 array of rotated keypoints.
        """
        flipped_keypoints = np.copy(keypoints)

        # Perform matrix multiplication to flip the keypoints
        flipped_keypoints = np.dot(flipped_keypoints, T.T)

        # Create the translation matrix
        T2 = np.array([[1, 0, center[0]],
                       [0, 1, center[1]],
                       [0, 0, 1]])


        # Add a third coordinate of 1 to the coordinates
        coordinates = flipped_keypoints[:, :2]
        coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))

        # Perform matrix multiplication to translate the coordinates
        translated_coordinates = np.dot(coordinates, T2.T)

        # Return the flippedkeypoint coordinates with the third coordinate removed
        return translated_coordinates[:, :2]

    def random_horizontal_flip(self, image, keypoints):
        if random.random() < 1: # 100% of images
            # Flip the image horizontally
            image = transforms.functional.hflip(image)
            # Flip the keypoints horizontally
            T = np.array([[-1, 0],
                          [ 0, 1]])
            center = [20/50.0,0]
            keypoints = self.flip_keypoints(keypoints, T, center)
        return image, keypoints

    def random_vertical_flip(self, image, keypoints):
        if random.random() < 1: #100% of images
            # Flip the image horizontally
            image = transforms.functional.vflip(image)
            # Flip the keypoints horizontally
            T = np.array([[1, 0],
                          [0,-1]])
            center = [0,20/50.0]
            keypoints = self.flip_keypoints(keypoints, T, center)
        return image, keypoints

    def random_rotation(self, image, keypoints):
        # Rotate the image by a random angle
        angle = random.uniform(-55, 55)
        # Rotate image and keypoints with same transform
        image = transforms.functional.rotate(image, angle)
        # Define the angle of rotation in radians
        theta = math.radians(angle*-1)

        # Create the rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        keypoints = self.rotate_keypoints(keypoints, R, center=[0,0])
        return image, keypoints

    # Define the transformation function
    def pre_processing(self, image, key_points):

        # Apply some random transformations to the image
        image, key_points = self.random_horizontal_flip(image, key_points)
        #image, key_points = self.random_rotation(image, key_points)
        #image, key_points = self.random_vertical_flip(image, key_points)
        return image, key_points

    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}