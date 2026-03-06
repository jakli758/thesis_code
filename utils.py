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