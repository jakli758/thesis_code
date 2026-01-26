import cv2
import numpy as np
from typing import Tuple
import os
from tqdm import tqdm

def rescale_img(img, size):
    """
    Change the size of the image to the given size. 
    
    :param img: image to be rescaled, must be squared
    :param size: lenght of the edges for the output image 
    """
    pass


def pad_img(img):
    """
    Add zero padding to the shorter side of the image to make it squared.
    
    :param img: image that should be padded
    """
    pass

def preprocess(path='/local/data1/jakli758/threeclasses/', new_shape=(128,128)):

    folders = ['accepted', 'rejected']
    
    for folder in folders:
        path_current = f'{path}{folder}'
        path_new = f'{path}{folder}_preprocessed_{new_shape[0]}'
        for file in tqdm(os.listdir(path_current)):
            

            # imread automatically converts img to 8-bit, if CV.IMREAD_ANYDEPTH is not set
            img = cv2.imread(f'{path_current}/{file}', 0)
            img_preprocessed = resize_with_pad(image=img,new_shape=new_shape, padding_color=0)
            
            cv2.imwrite(f'{path_new}/{file}', img_preprocessed)

    
# https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (0, 0, 0)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image



if __name__== "__main__":
    pass