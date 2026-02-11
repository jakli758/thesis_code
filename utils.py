import cv2
import numpy as np
from typing import Tuple
import os
from tqdm import tqdm
import os
import re
import shutil
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

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


def grey_to_RGB(path='/local/data1/jakli758/threeclasses/', img_size=128):
    folders = [f'accepted_preprocessed_{img_size}', f'rejected_preprocessed_{img_size}']
    
    for folder in folders:
        path_current = f'{path}{folder}'
        path_new = f'{path}{folder}_rgb'
        for file in tqdm(os.listdir(path_current)):
            img = cv2.imread(f'{path_current}/{file}', 0)
            img_preprocessed = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            cv2.imwrite(f'{path_new}/{file}', img_preprocessed)

def datasplit_dm(path='/local/data1/jakli758/', img_size=128):
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT = f'{path}threeclasses/accepted_preprocessed_{img_size}_rgb/' 
    OUTPUT_ROOT = f'{path}dataset/dm/{img_size}/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT):
        for fname in os.listdir(SOURCE_ROOT):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    patients = list(patient_class.keys())
    labels = [patient_class[p] for p in patients]

    train_patients, test_patients, _, _ = train_test_split(
        patients,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    train_patients = set(train_patients)
    test_patients  = set(test_patients)
    

    for split in ['train', 'test']:
        for cls in ['AFF', 'CONTROL']:
            os.makedirs(f'{OUTPUT_ROOT}{split}/{cls}', exist_ok=True)

    for pid in train_patients:
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/train/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)

    for pid in test_patients:
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/test/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)


    def count_by_class(patient_set):
        return Counter(patient_class[p] for p in patient_set)

    print("\nPatient distribution:")
    print("Train:", count_by_class(train_patients))
    print("Test: ", count_by_class(test_patients))

    assert train_patients.isdisjoint(test_patients), "Data leakage detected!"

    print("\nDataset successfully created at:", OUTPUT_ROOT)



def datasplit_gan(path='/local/data1/jakli758/', img_size=128):
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT = f'{path}threeclasses/accepted_preprocessed_{img_size}_rgb/' 
    OUTPUT_ROOT = f'{path}dataset/gan/{img_size}/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT):
        for fname in os.listdir(SOURCE_ROOT):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    patients = list(patient_class.keys())
    labels = [patient_class[p] for p in patients]

    train_patients, test_patients, _, _ = train_test_split(
        patients,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    train_patients = set(train_patients)
    test_patients  = set(test_patients)
    

    for split in ['trainA', 'trainB', 'testA', 'testB']:
        os.makedirs(f'{OUTPUT_ROOT}{split}', exist_ok=True)

    for pid in train_patients:
        for cls, filepath in patient_files[pid]:
            if cls == 'AFF':
                dst = f'{OUTPUT_ROOT}/trainA/{os.path.basename(filepath)}'
            elif cls == 'CONTROL':
                dst = f'{OUTPUT_ROOT}/trainB/{os.path.basename(filepath)}'
                
            shutil.copy2(filepath, dst)

    for pid in test_patients:
        for cls, filepath in patient_files[pid]:
            if cls == 'AFF':
                dst = f'{OUTPUT_ROOT}/testA/{os.path.basename(filepath)}'
            elif cls == 'CONTROL':
                dst = f'{OUTPUT_ROOT}/testB/{os.path.basename(filepath)}'
                
            shutil.copy2(filepath, dst)
            shutil.copy2(filepath, dst)


    def count_files_by_class(split):
        split_dir = os.path.join(OUTPUT_ROOT, split)
        counts = {}
        for cls in ["A", "B"]:
            class_dir = f'{split_dir}{cls}'
            counts[cls] = len([
                f for f in os.listdir(class_dir) 
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
        return counts

    print("\nFile distribution per class:")
    print("Train:", count_files_by_class("train"))
    print("Test: ", count_files_by_class("test"))

    # Double check patient leakage
    assert train_patients.isdisjoint(test_patients), "Data leakage detected!"

    print("\nDataset successfully created at:", OUTPUT_ROOT)
