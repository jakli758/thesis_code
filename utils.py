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


def count_files_per_split(dataset_root):
    splits = ['train', 'val', 'test']
    classes = ['AFF', 'CONTROL', 'HEALTHY']

    results = defaultdict(dict)

    total_all = 0

    for split in splits:
        split_total = 0
        for cls in classes:
            class_path = os.path.join(dataset_root, split, cls)

            if os.path.exists(class_path):
                n_files = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith('.png')
                ])
            else:
                n_files = 0

            results[split][cls] = n_files
            split_total += n_files

        results[split]['TOTAL'] = split_total
        total_all += split_total

    # Pretty print
    print("\nFile distribution per split:\n")
    for split in splits:
        print(f"{split.upper()}:")
        for cls in classes:
            print(f"  {cls:<8}: {results[split][cls]}")
        print(f"  TOTAL   : {results[split]['TOTAL']}\n")

    print(f"GRAND TOTAL: {total_all}\n")


def count_files_per_split_gan(dataset_root):
    """Count files per split for GAN-style dataset structure.
    
    Works with flat directory structure: train_AFF, train_NFF, train_HEALTHY, 
    test_AFF, test_NFF, test_HEALTHY, left_out_AFF, left_out_NFF, left_out_HEALTHY
    """
    splits = ['train', 'test', 'left_out']
    class_mapping = {'AFF': 'AFF', 'NFF': 'CONTROL', 'HEALTHY': 'HEALTHY'}
    
    results = defaultdict(dict)
    total_all = 0
    
    for split in splits:
        split_total = 0
        for class_abbrev, class_name in class_mapping.items():
            dir_name = f'{split}_{class_abbrev}'
            class_path = os.path.join(dataset_root, dir_name)
            
            if os.path.exists(class_path):
                n_files = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith('.png')
                ])
            else:
                n_files = 0
            
            results[split][class_name] = n_files
            split_total += n_files
        
        results[split]['TOTAL'] = split_total
        total_all += split_total
    
    # Pretty print
    print("\nFile distribution per split (GAN):\n")
    for split in splits:
        print(f"{split.upper()}:")
        for class_name in class_mapping.values():
            print(f"  {class_name:<8}: {results[split][class_name]}")
        print(f"  TOTAL   : {results[split]['TOTAL']}\n")
    
    print(f"GRAND TOTAL: {total_all}\n")


def datasplit_dm(path='/local/data1/jakli758/', img_size=128):
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT_FRAC = f'{path}threeclasses/accepted_preprocessed_{img_size}_rgb/' 
    SOURCE_ROOT_HEL = f'{path}threeclasses/rejected_preprocessed_{img_size}_rgb/' 
    OUTPUT_ROOT = f'{path}dataset_new/dm/{img_size}/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT_HEL):
        for fname in os.listdir(SOURCE_ROOT_HEL):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT_HEL, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    if os.path.exists(SOURCE_ROOT_FRAC):
        for fname in os.listdir(SOURCE_ROOT_FRAC):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT_FRAC, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    patients = list(patient_class.keys())
    labels = [patient_class[p] for p in patients]

    train_val_patients, test_patients, train_val_labels, _ = train_test_split(
        patients,
        labels,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )

    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients,
        train_val_labels,
        test_size=0.10,
        random_state=42,
        stratify=train_val_labels
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)
    test_patients  = set(test_patients)
    

    for split in ['train', 'val', 'test']:
        for cls in ['AFF', 'CONTROL', 'HEALTHY']:
            os.makedirs(f'{OUTPUT_ROOT}{split}/{cls}', exist_ok=True)

    for pid in train_patients:
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/train/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)
    
    for pid in val_patients:
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/val/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)

    for pid in test_patients:
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/test/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)


    def count_by_class(patient_set):
        return Counter(patient_class[p] for p in patient_set)

    print("\nPatient distribution:")
    print("Train:", count_by_class(train_patients))
    print("Val:", count_by_class(val_patients))
    print("Test: ", count_by_class(test_patients))

    count_files_per_split(OUTPUT_ROOT)

    assert train_patients.isdisjoint(test_patients), "Data leakage detected!"

    print("\nDataset successfully created at:", OUTPUT_ROOT)



def datasplit_gan(path='/local/data1/jakli758/', img_size=128):
    print("Start datasplit GAN")
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT_FRAC = f'{path}threeclasses/accepted_preprocessed_{img_size}_rgb/' 
    SOURCE_ROOT_HEL = f'{path}threeclasses/rejected_preprocessed_{img_size}_rgb/' 
    OUTPUT_ROOT = f'{path}dataset_new/gan/{img_size}/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT_HEL):
        for fname in os.listdir(SOURCE_ROOT_HEL):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT_HEL, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    if os.path.exists(SOURCE_ROOT_FRAC):
        for fname in os.listdir(SOURCE_ROOT_FRAC):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = m.group(2)

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT_FRAC, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    patients = list(patient_class.keys())
    labels = [patient_class[p] for p in patients]

    train_val_patients, test_patients, train_val_labels, _ = train_test_split(
        patients,
        labels,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )

    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients,
        train_val_labels,
        test_size=0.10,
        random_state=42,
        stratify=train_val_labels
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)
    test_patients  = set(test_patients)
    
    # create directories for each class split
    for split in ['train_AFF', 'train_NFF', 'train_HEALTHY', 'test_AFF', 'test_NFF', 'test_HEALTHY', 'left_out_AFF', 'left_out_NFF', 'left_out_HEALTHY']:
        os.makedirs(f'{OUTPUT_ROOT}{split}', exist_ok=True)

    for pid in train_patients:
        for cls, filepath in patient_files[pid]:
            if cls == 'AFF':
                dst = f'{OUTPUT_ROOT}/train_AFF/{os.path.basename(filepath)}'
            elif cls == 'CONTROL':
                dst = f'{OUTPUT_ROOT}/train_NFF/{os.path.basename(filepath)}'
            else:
                dst = f'{OUTPUT_ROOT}/train_HEALTHY/{os.path.basename(filepath)}'
                
            shutil.copy2(filepath, dst)
        
    for pid in val_patients:
        for cls, filepath in patient_files[pid]:
            if cls == 'AFF':
                dst = f'{OUTPUT_ROOT}/test_AFF/{os.path.basename(filepath)}'
            elif cls == 'CONTROL':
                dst = f'{OUTPUT_ROOT}/test_NFF/{os.path.basename(filepath)}'
            else:
                dst = f'{OUTPUT_ROOT}/test_HEALTHY/{os.path.basename(filepath)}'
                
            shutil.copy2(filepath, dst)
            
    
    
    for pid in test_patients:
        for cls, filepath in patient_files[pid]:
            if cls == 'AFF':
                dst = f'{OUTPUT_ROOT}/left_out_AFF/{os.path.basename(filepath)}'
            elif cls == 'CONTROL':
                dst = f'{OUTPUT_ROOT}/left_out_NFF/{os.path.basename(filepath)}'
            else:
                dst = f'{OUTPUT_ROOT}/left_out_HEALTHY/{os.path.basename(filepath)}'
                
            shutil.copy2(filepath, dst)


    def count_by_class(patient_set):
        return Counter(patient_class[p] for p in patient_set)

    print("\nPatient distribution:")
    print("Train:", count_by_class(train_patients))
    print("Val:", count_by_class(val_patients))
    print("Test: ", count_by_class(test_patients))

    count_files_per_split_gan(OUTPUT_ROOT)

    # Double check patient leakage
    assert train_patients.isdisjoint(test_patients), "Data leakage detected!"

    print("\nDataset successfully created at:", OUTPUT_ROOT)


