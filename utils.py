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
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd



def plot_img(path, is_rgb=True):
    if is_rgb:
        img = cv2.imread(path)
        plt.imshow(img)
    else:
        img = cv2.imread(path, 0)
        plt.imshow(img, cmap="gray")

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

def preprocess(path='/local/data1/jakli758/threeclasses/', new_path="/local/data1/jakli758/threeclasses/", new_shape=(128,128)):

    folders = ['accepted', 'rejected']
    
    for folder in folders:
        
        path_current = f'{path}{folder}'
        path_new = f"{new_path}{folder}"
        os.makedirs(path_new, exist_ok=True)
        for file in tqdm(os.listdir(path_current)):
            
            img = cv2.imread(f'{path_current}/{file}', -1)
            img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            # check if image has white background. If so, invert image
            if has_bright_background(img_8bit):
                img_8bit = 255 - img_8bit

            img_preprocessed = resize_with_pad(img_8bit,new_shape=new_shape, padding_color=(0,0,0))
            plt.imsave(os.path.join(path_new,file), img_preprocessed, cmap='gray')

def has_bright_background(img, border_frac=0.08, threshold=127):
    h, w = img.shape
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    top = img[:bh, :]
    bottom = img[-bh:, :]
    left = img[:, :bw]
    right = img[:, -bw:]

    border_pixels = np.concatenate([
        top.ravel(), bottom.ravel(), left.ravel(), right.ravel()
    ])

    return np.median(border_pixels) > threshold


    
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
                fracture_type = "HEALTHY"

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
    SOURCE_ROOT_FRAC = f'{path}dataset_uint8_inverted/accepted/' 
    SOURCE_ROOT_HEL = f'{path}dataset_uint8_inverted/rejected/' 
    OUTPUT_ROOT = f'{path}dataset_uint8_inverted/split_gan/{img_size}/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT_HEL):
        for fname in os.listdir(SOURCE_ROOT_HEL):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = "HEALTHY"

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


def datasplit_gan_raw(path='/local/data1/jakli758/'):
    print("Start datasplit GAN")
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT_FRAC = f'{path}threeclasses/accepted/' 
    SOURCE_ROOT_HEL = f'{path}threeclasses/rejected/' 
    OUTPUT_ROOT = f'{path}dataset_normalized/gan/raw/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    if os.path.exists(SOURCE_ROOT_HEL):
        for fname in os.listdir(SOURCE_ROOT_HEL):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = "HEALTHY"

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


def datasplit_dm_raw(path='/local/data1/jakli758/'):
    PATTERN = re.compile(r'patient_(\d*)_([A-Z]*)')
    SOURCE_ROOT_FRAC = f'{path}threeclasses/accepted/' 
    SOURCE_ROOT_HEL = f'{path}threeclasses/rejected/' 
    OUTPUT_ROOT = f'{path}dataset_normalized/dm/raw/'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # keys: patient_nr; values: lists containing tuples (fracture_type, file_name)
    patient_files = defaultdict(list)

    patient_class = defaultdict(list)

    print("Loading images")
    if os.path.exists(SOURCE_ROOT_HEL):
        for fname in tqdm(os.listdir(SOURCE_ROOT_HEL)):
            if fname.lower().endswith((".png")):
                m = PATTERN.search(fname)
                pid = m.group(1)
                fracture_type = "HEALTHY"

                patient_files[pid].append(
                    (fracture_type, os.path.join(SOURCE_ROOT_HEL, fname))
                )

                # takes last fracture type as class for this patient, even if they had different ones before
                patient_class[pid] = fracture_type
            else:
                print(f'Not an image file: {fname}')

    if os.path.exists(SOURCE_ROOT_FRAC):
        for fname in tqdm(os.listdir(SOURCE_ROOT_FRAC)):
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

    print("Performing datasplit")

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

    print("Copying files to target directory")

    for pid in tqdm(train_patients):
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/train/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)
    
    for pid in tqdm(val_patients):
        for cls, filepath in patient_files[pid]:
            dst = f'{OUTPUT_ROOT}/val/{cls}/{os.path.basename(filepath)}'
            shutil.copy2(filepath, dst)

    for pid in tqdm(test_patients):
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


def rotate_and_save(image_path):
    try:
        img = Image.open(image_path)
        base_name, ext = os.path.splitext(image_path)

        rotations = {
            90: "_rot90",
            180: "_rot180",
            270: "_rot270"
        }

        for angle, suffix in rotations.items():
            rotated = img.rotate(angle, expand=True)
            new_filename = f"{base_name}{suffix}{ext}"
            rotated.save(new_filename)

        print(f"Processed: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def augment_dataset(root_dir):
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            print(file)
            if file.lower().endswith(VALID_EXTENSIONS):
                image_path = os.path.join(root, file)

                # Avoid re-augmenting already rotated images
                if "_rot90" in file or "_rot180" in file or "_rot270" in file:
                    continue
                rotate_and_save(image_path)


def plot_intensity_histogram(folder_path):
    bins = 256  # Number of intensity bins

    # ---- LOAD IMAGES AND COMPUTE HISTOGRAMS ----
    all_histograms = []
    combined_histogram = np.zeros(bins)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Read image
        image = cv2.imread(file_path)

        if image is None:
            continue  # Skip non-image files

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = hist.flatten()

        # Normalize (optional but recommended)
        hist = hist / hist.sum()

        all_histograms.append((filename, hist))
        combined_histogram += hist

        # Normalize combined histogram
        combined_histogram = combined_histogram / combined_histogram.sum()

    # ---- PLOT 1: Combined Histogram ----
    plt.figure()
    plt.plot(combined_histogram)
    plt.title("Combined Intensity Distribution (All Images)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.xlim([0, 255])
    plt.show()

    # ---- PLOT 2: One Curve Per Image ----
    plt.figure()

    for filename, hist in all_histograms:
        plt.plot(hist, label=filename)

    plt.title("Intensity Distribution Per Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.xlim([0, 255])
    plt.show()




#######################


# def load_grayscale(path: str) -> np.ndarray | None:
#     """Load an image as grayscale. Supports common formats; returns None if unreadable."""
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         return None
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img

# def to_uint8(img: np.ndarray, low_pct=1, high_pct=99) -> np.ndarray:
#     """Clip to percentiles and rescale to uint8 [0,255]. Works for uint8/uint16/float."""
#     x = img.astype(np.float32)

#     lo = np.percentile(x, low_pct)
#     hi = np.percentile(x, high_pct)
#     if hi <= lo:
#         # fallback: just min/max
#         lo, hi = float(x.min()), float(x.max())
#         if hi <= lo:
#             return np.zeros_like(img, dtype=np.uint8)

#     x = np.clip(x, lo, hi)
#     x = (x - lo) / (hi - lo)  # 0..1
#     x = (255.0 * x).round().astype(np.uint8)
#     return x


# def make_foreground_mask(img_u8):
#     """
#     Returns a boolean mask of the foreground (leg/bone region).
#     Assumes background is darker than foreground (typical X-ray exports).
#     """

#     # Foreground mask settings
#     # If your background is black-ish, Otsu works well.
#     USE_OTSU = True
#     FIXED_THRESH = 10  # used only if USE_OTSU=False (0..255), tweak if needed
#     MORPH_KERNEL = 7   # size for morphological cleanup; try 5,7,9
#     MIN_FG_FRAC = 0.01 # if mask is too tiny, fall back to whole image
    
#     if USE_OTSU:
#         # Otsu threshold; foreground = above threshold
#         _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, th = cv2.threshold(img_u8, FIXED_THRESH, 255, cv2.THRESH_BINARY)

#     # Morphological cleanup: close holes, remove speckles
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
#     th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)

#     mask = th.astype(bool)

#     # If mask is tiny or empty, fall back to whole image
#     if mask.mean() < MIN_FG_FRAC:
#         mask = np.ones_like(mask, dtype=bool)

#     return mask


# def robust_zscore_with_mask(img, mask, eps=1e-8):
#     """
#     Robust z-score using median + IQR computed only on masked pixels.
#     z = (x - median) / (IQR/1.349)
#     """
#     x = img.astype(np.float32)
#     vals = x[mask]

#     if vals.size == 0:
#         return np.zeros_like(x, dtype=np.float32)

#     med = float(np.median(vals))
#     q1 = float(np.percentile(vals, 25))
#     q3 = float(np.percentile(vals, 75))
#     iqr = q3 - q1

#     print("Calculated statisics:\n")
#     print(f"Median: {med}\nq1: {q1}\nq2:{q3}\nIQR:{iqr}")

#     # Convert IQR to a "robust std" estimate (Gaussian assumption)
#     robust_sigma = iqr / 1.349 if iqr > eps else 0.0
#     if robust_sigma < eps:
#         return np.zeros_like(x, dtype=np.float32)

#     z = (x - med) / robust_sigma
#     return z

# def normalize_and_plot_hist(input_folder, output_folder):


#     # -----------------------
#     # Settings
#     # -----------------------
#     os.makedirs(output_folder, exist_ok=True)

#     bins_before = 256
#     bins_after = 256              # AFTER is vis uint8, so 256 bins is natural
#     z_clip = 5.0
#     alpha_per_image = 0.25

#     save_visualization_png = True  # you said you only need this

#     # -----------------------
#     # Hist helpers (uint8)
#     # -----------------------
#     def hist_u8(img_u8, bins=256):
#         h = cv2.calcHist([img_u8], [0], None, [bins], [0, 256]).flatten()
#         s = h.sum()
#         return h / s if s > 0 else h


#     # -----------------------
#     # Process folder
#     # -----------------------
#     names = []
#     raw_u8_images = []
#     vis_u8_images = []   # <-- store vis images for "after" histograms

#     for fn in tqdm(sorted(os.listdir(input_folder))):
#         in_path = os.path.join(input_folder, fn)

#         # Skip directories
#         if not os.path.isfile(in_path):
#             continue

#         img = load_grayscale(in_path)
#         if img is None:
#             continue

#         img_u8 = to_uint8(img)
#         mask = make_foreground_mask(img_u8)

#         # Robust z-score (float32), used only to create vis
#         z_img = robust_zscore_with_mask(img, mask)

#         # Visualization PNG (uint8 0..255)
#         # This is what you keep + save.
#         vis = np.clip(z_img, -z_clip, z_clip)
#         vis = (vis + z_clip) / (2 * z_clip)   # -> 0..1
#         vis = (255 * vis).round().astype(np.uint8)

#         names.append(fn)
#         raw_u8_images.append(img_u8)
#         vis_u8_images.append(vis)

#         if save_visualization_png:
#             base = os.path.splitext(fn)[0]
#             out_path = os.path.join(output_folder, base + "_vis.png")
#             ok = cv2.imwrite(out_path, vis)
#             if not ok:
#                 print(f"Warning: failed to write {out_path}")

#     print(f"Saved {len(vis_u8_images)} visualization images to: {output_folder}")

#     if len(vis_u8_images) == 0:
#         raise RuntimeError("No images processed. Check input_folder and file formats.")


#     # -----------------------
#     # Histograms before/after
#     # -----------------------
#     raw_hists = [hist_u8(im, bins_before) for im in raw_u8_images]
#     vis_hists = [hist_u8(im, bins_after) for im in vis_u8_images]

#     combined_raw = np.mean(np.stack(raw_hists, axis=0), axis=0) if raw_hists else np.zeros(bins_before)
#     combined_vis = np.mean(np.stack(vis_hists, axis=0), axis=0) if vis_hists else np.zeros(bins_after)

#     x_u8 = np.arange(256)

#     # Combined (before vs after)
#     plt.figure()
#     plt.plot(x_u8, combined_raw, label="Before (raw uint8)")
#     plt.plot(x_u8, combined_vis, label="After (vis uint8)")
#     plt.title("Combined Intensity Distribution - BEFORE vs AFTER (vis)")
#     plt.xlabel("Pixel intensity (0..255)")
#     plt.ylabel("Normalized frequency")
#     plt.xlim([0, 255])
#     plt.legend()
#     plt.show()

#     # Per-image curves (before)
#     plt.figure()
#     for h in raw_hists:
#         plt.plot(x_u8, h, alpha=alpha_per_image)
#     plt.title("Per-Image Histograms - BEFORE")
#     plt.xlabel("Pixel intensity (0..255)")
#     plt.ylabel("Normalized frequency")
#     plt.xlim([0, 255])
#     plt.show()

#     # Per-image curves (after)
#     plt.figure()
#     for h in vis_hists:
#         plt.plot(x_u8, h, alpha=alpha_per_image)
#     plt.title("Per-Image Histograms - AFTER (vis)")
#     plt.xlabel("Pixel intensity (0..255)")
#     plt.ylabel("Normalized frequency")
#     plt.xlim([0, 255])
#     plt.show()



def norm_dataset_level(input_folder, output_folder):

    # -----------------------
    # Settings
    # -----------------------
    os.makedirs(output_folder, exist_ok=True)

    # Hist / visualization
    bins_u8 = 256
    alpha_per_image = 0.25
    z_clip = 5.0
    save_visualization_png = True

    # Foreground mask settings (assumes background mostly dark)
    USE_OTSU = True
    FIXED_THRESH = 10          # only used if USE_OTSU=False
    MORPH_KERNEL = 7
    MIN_FG_FRAC = 0.01

    # Dataset-stat estimation settings
    # Use a maximum number of pixels per image for speed/memory
    MAX_SAMPLES_PER_IMAGE = 200_000
    RANDOM_SEED = 0

    # -----------------------
    # Helpers
    # -----------------------
    def load_grayscale(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def to_uint8(img):
        """Create uint8 version for masking/hist-before (supports uint16)."""
        if img.dtype == np.uint8:
            return img
        x = img.astype(np.float32)
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
        x = (x - lo) / (hi - lo)
        return (255.0 * x).round().astype(np.uint8)

    def make_foreground_mask(img_u8):
        if USE_OTSU:
            _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(img_u8, FIXED_THRESH, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)

        mask = th.astype(bool)
        if mask.mean() < MIN_FG_FRAC:
            mask = np.ones_like(mask, dtype=bool)
        return mask

    def robust_sigma_from_iqr(vals, eps=1e-8):
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1
        if iqr <= eps:
            return 0.0
        return iqr / 1.349  # Gaussian-consistent scale

    def hist_u8(img_u8, bins=256):
        h = cv2.calcHist([img_u8], [0], None, [bins], [0, 256]).flatten()
        s = h.sum()
        return h / s if s > 0 else h

    # -----------------------
    # Collect file list
    # -----------------------
    paths = []
    names = []
    for fn in sorted(os.listdir(input_folder)):
        p = os.path.join(input_folder, fn)
        if os.path.isfile(p):
            paths.append(p)
            names.append(fn)

    if not paths:
        raise RuntimeError("No files found in input_folder.")

    # -----------------------
    # PASS 1: Estimate dataset-level stats (median + robust sigma) from sampled foreground pixels
    # -----------------------
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []

    for p, fn in tqdm(list(zip(paths, names)), desc="Estimating dataset stats"):
        img = load_grayscale(p)
        if img is None:
            continue

        img_u8 = to_uint8(img)
        mask = make_foreground_mask(img_u8)

        x = img.astype(np.float32)
        vals = x[mask]

        if vals.size == 0:
            continue

        # Randomly subsample to limit memory
        if vals.size > MAX_SAMPLES_PER_IMAGE:
            idx = rng.choice(vals.size, size=MAX_SAMPLES_PER_IMAGE, replace=False)
            vals = vals[idx]

        samples.append(vals)

    if not samples:
        raise RuntimeError("Failed to collect samples for dataset stats (all images unreadable or masks empty).")

    all_vals = np.concatenate(samples, axis=0)

    dataset_median = float(np.median(all_vals))
    dataset_sigma = float(robust_sigma_from_iqr(all_vals))

    if dataset_sigma <= 1e-8:
        raise RuntimeError("Dataset robust sigma is ~0. Check your data/masks; intensities may be constant.")

    print(f"Dataset-level median: {dataset_median:.4f}")
    print(f"Dataset-level robust sigma (IQR/1.349): {dataset_sigma:.4f}")
    print(f"Total sampled pixels: {all_vals.size}")

    # -----------------------
    # PASS 2: Apply dataset-level normalization, save vis PNGs, compute histograms
    # -----------------------
    raw_u8_images = []
    vis_u8_images = []

    for p, fn in tqdm(list(zip(paths, names)), desc="Normalizing + saving"):
        img = load_grayscale(p)
        if img is None:
            print(f"Skipping unreadable: {fn}")
            continue

        img_u8 = to_uint8(img)

        # Dataset-level robust z-score (no per-image stats)
        x = img.astype(np.float32)
        z = (x - dataset_median) / dataset_sigma

        # Convert to vis uint8
        vis = np.clip(z, -z_clip, z_clip)
        vis = (vis + z_clip) / (2 * z_clip)      # -> 0..1
        vis = (255 * vis).round().astype(np.uint8)

        raw_u8_images.append(img_u8)
        vis_u8_images.append(vis)

        if save_visualization_png:
            base = os.path.splitext(fn)[0]
            out_path = os.path.join(output_folder, base + ".png")
            ok = cv2.imwrite(out_path, vis)
            if not ok:
                print(f"Warning: failed to write {out_path}")

    print(f"Saved {len(vis_u8_images)} normalized visualization PNGs to: {output_folder}")

    if len(vis_u8_images) == 0:
        raise RuntimeError("No images processed in normalization pass.")

    # -----------------------
    # Histograms before/after (both are uint8 here)
    # -----------------------
    raw_hists = [hist_u8(im, bins_u8) for im in raw_u8_images]
    vis_hists = [hist_u8(im, bins_u8) for im in vis_u8_images]

    combined_raw = np.mean(np.stack(raw_hists, axis=0), axis=0) if raw_hists else np.zeros(bins_u8)
    combined_vis = np.mean(np.stack(vis_hists, axis=0), axis=0) if vis_hists else np.zeros(bins_u8)

    x_u8 = np.arange(256)

    plt.figure()
    plt.plot(x_u8, combined_raw, label="Raw")
    plt.plot(x_u8, combined_vis, label="Normalized")
    plt.title("Combined Intensity Distribution")
    plt.xlabel("Pixel intensity (0..255)")
    plt.ylabel("Normalized frequency")
    plt.xlim([0, 255])
    plt.legend()
    plt.show()

    plt.figure()
    for h in raw_hists:
        plt.plot(x_u8, h, alpha=alpha_per_image)
    plt.title("Per-Image Histograms - Raw")
    plt.xlabel("Pixel intensity (0..255)")
    plt.ylabel("Normalized frequency")
    plt.xlim([0, 255])
    plt.show()

    plt.figure()
    for h in vis_hists:
        plt.plot(x_u8, h, alpha=alpha_per_image)
    plt.title("Per-Image Histograms - Normalized")
    plt.xlabel("Pixel intensity (0..255)")
    plt.ylabel("Normalized frequency")
    plt.xlim([0, 255])
    plt.show()





def normalize_domain_level(input_folder, output_folder):
    """
    Computes global mean and std over all RGB images in input_folder
    and normalizes them using:

        normalized_image = (image - global_mean) / global_std + 128

    The normalized images are saved to output_folder using plt.imsave.
    """

    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    if len(image_files) == 0:
        raise ValueError("No images found in input folder.")

    # -------- First pass: compute global mean and std --------
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for fname in tqdm(image_files):
        path = os.path.join(input_folder, fname)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        pixel_count += img.size

    global_mean = pixel_sum / pixel_count
    global_std = np.sqrt(pixel_sq_sum / pixel_count - global_mean ** 2)

    print(f"Global mean: {global_mean}")
    print(f"Global std: {global_std}")

    # -------- Second pass: normalize and save --------
    for fname in tqdm(image_files):
        path = os.path.join(input_folder, fname)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        normalized = (img - global_mean) / global_std + 128
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        out_path = os.path.join(output_folder, fname)
        plt.imsave(out_path, normalized)

    print("Normalization finished.")





def count_images_per_hospital(directory):
    """
    Reads filenames in a directory and counts images per hospital,
    split by AFF and NFF.

    Returns:
        dict: {hospital_id: {'AFF': count, 'NFF': count}}
    """

    hospital_counts = defaultdict(lambda: {'AFF': 0, 'NFF': 0})

    for filename in os.listdir(directory):
        if not filename.endswith(".png"):
            continue

        # Extract hospital number
        hospital_match = re.search(r"hospital_(\d+)", filename)

        # Extract label
        label_match = re.search(r"_(AFF|NFF)_", filename)

        if hospital_match and label_match:
            hospital = int(hospital_match.group(1))
            label = label_match.group(1)

            hospital_counts[hospital][label] += 1

    return dict(hospital_counts)


def plot_hospital_distribution(hospital_dict):

    hospitals = sorted(hospital_dict.keys())
    aff_counts = [hospital_dict[h]['AFF'] for h in hospitals]
    nff_counts = [hospital_dict[h]['NFF'] for h in hospitals]

    totals = [a + b for a, b in zip(aff_counts, nff_counts)]

    x = np.arange(len(hospitals))

    plt.figure(figsize=(10,5))

    plt.bar(x, aff_counts, label="AFF")
    plt.bar(x, nff_counts, bottom=aff_counts, label="NFF")

    # Write totals above bars
    for i, total in enumerate(totals):
        plt.text(
            x[i],
            total + max(totals)*0.01,  # small offset
            str(total),
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.xlabel("Hospital ID")
    plt.ylabel("Number of Images")
    plt.title("Images per Hospital")

    max_labels = 100
    if len(hospitals) > max_labels:
        step = len(hospitals) // max_labels + 1
        ticks = x[::step]
        labels = [hospitals[i] for i in range(0, len(hospitals), step)]
    else:
        ticks = x
        labels = hospitals

    plt.xticks(ticks, labels, rotation=90, ha="right")

    plt.legend()
    plt.tight_layout()
    plt.show()






def plot_folder_histograms(
    folder_path: str,
    pattern: str = "*",
    grayscale: bool = True,
    bins: int = 256,
    density: bool = True,
    alpha: float = 0.25,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot two histograms for images in a folder:
    1) A single histogram over all pixels from all images combined
    2) Overlapping histograms, one per image

    Parameters
    ----------
    folder_path : str
        Path to the folder containing images.
    pattern : str
        Glob pattern to filter files, e.g. "*.png" or "*.jpg".
        Default "*" tries all files and filters valid images internally.
    grayscale : bool
        If True, convert images to grayscale before computing histograms.
        If False, uses all RGB values flattened together.
    bins : int
        Number of histogram bins.
    density : bool
        If True, normalize histograms to compare distributions.
    alpha : float
        Transparency for per-image overlapping histograms.
    figsize : tuple
        Figure size for the 2-panel plot.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.glob(pattern) if p.suffix.lower() in image_extensions]

    if not files:
        raise ValueError(f"No supported image files found in: {folder}")

    all_pixels = []
    per_image_pixels = []
    labels = []

    for file in sorted(files):
        try:
            img = Image.open(file)
            img = img.convert("L" if grayscale else "RGB")
            arr = np.asarray(img).ravel()

            all_pixels.append(arr)
            per_image_pixels.append(arr)
            labels.append(file.name)
        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    if not per_image_pixels:
        raise ValueError("No readable images found.")

    all_pixels = np.concatenate(all_pixels)

    value_range = (0, 255)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram over all images combined
    axes[0].hist(
        all_pixels,
        bins=bins,
        range=value_range,
        density=density,
        alpha=0.9,
    )
    axes[0].set_title("Histogram of all images combined")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Density" if density else "Count")

    # Overlapping histogram per image
    for arr, label in zip(per_image_pixels, labels):
        axes[1].hist(
            arr,
            bins=bins,
            range=value_range,
            density=density,
            alpha=alpha,
            label=label,
        )

    axes[1].set_title("Overlapping histogram per image")
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Density" if density else "Count")

    # Only show legend if there aren't too many images
    if len(labels) <= 12:
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()






def parse_cyclegan_losses(log_file):
    """
    Parse a CycleGAN training log file and extract loss values.

    Parameters
    ----------
    log_file : str or Path
        Path to the training log text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['epoch', 'iters', 'log_index', 'D_A', 'G_A', 'cycle_A', 'idt_A',
         'D_B', 'G_B', 'cycle_B', 'idt_B']
    """
    log_file = Path(log_file)

    # Match lines like:
    # [Rank 0] (epoch: 1, iters: 200, time: 0.090, data: 0.108) , D_A: 0.394, ...
    pattern = re.compile(
        r"\(epoch:\s*(?P<epoch>\d+),\s*iters:\s*(?P<iters>\d+),.*?\)\s*,\s*"
        r"D_A:\s*(?P<D_A>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"G_A:\s*(?P<G_A>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"cycle_A:\s*(?P<cycle_A>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"idt_A:\s*(?P<idt_A>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"D_B:\s*(?P<D_B>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"G_B:\s*(?P<G_B>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"cycle_B:\s*(?P<cycle_B>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
        r"idt_B:\s*(?P<idt_B>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )

    LOSS_NAMES = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]

    rows = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                row = {
                    "epoch": int(match.group("epoch")),
                    "iters": int(match.group("iters")),
                }
                for loss in LOSS_NAMES:
                    row[loss] = float(match.group(loss))
                rows.append(row)

    if not rows:
        raise ValueError(f"No loss entries found in file: {log_file}")

    df = pd.DataFrame(rows)
    df["log_index"] = range(1, len(df) + 1)
    return df


def plot_cyclegan_losses(log_file, smooth_window=None, figsize=(16, 8)):
    """
    Parse a CycleGAN log file and plot the losses.

    Parameters
    ----------
    log_file : str or Path
        Path to the training log text file.
    smooth_window : int or None, optional
        Rolling average window size for smoothing. Example: 5.
        If None, raw curves are plotted.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    pd.DataFrame
        Parsed loss DataFrame.
    """
    df = parse_cyclegan_losses(log_file)
    LOSS_NAMES = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]

    plot_df = df.copy()
    if smooth_window is not None and smooth_window > 1:
        for loss in LOSS_NAMES:
            plot_df[loss] = plot_df[loss].rolling(window=smooth_window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, loss in zip(axes, LOSS_NAMES):
        ax.plot(df["log_index"], df[loss], label="raw")
        if smooth_window is not None and smooth_window > 1:
            ax.plot(plot_df["log_index"], plot_df[loss], label=f"smoothed ({smooth_window})")
            ax.legend()
        ax.set_title(loss)
        ax.set_xlabel("Log entry")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()