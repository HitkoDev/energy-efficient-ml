import cv2
import numpy as np
import pandas as pd
import os
from skimage import color, feature, measure, filters

def calculate_contrast(img):
    return img.std()

# https://stackoverflow.com/a/3498247
def calculate_perceived_brightness(img, method='avg'):
    if img.ndim == 2 or img.shape[2] == 1:  # If the image is already grayscale
        r = g = b = img
    else:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    if method == 'avg':
        return np.mean(0.299 * r + 0.587 * g + 0.114 * b)
    elif method == 'rms':
        return np.sqrt(np.mean((0.299 * r + 0.587 * g + 0.114 * b) ** 2))
    else:
        raise ValueError("Unsupported method")

def calculate_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def calculate_edges_new(img):
    edges = feature.canny(color.rgb2gray(img), sigma=3)
    contours = measure.find_contours(edges, fully_connected='high')

     # Check if any contours are found
    if len(contours) == 0:
        return np.zeros(7)  # Return an empty histogram if no contours are found

    # Calculating the length of each contour
    contour_lengths = [np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))) for contour in contours]
    edge_histogram, _ = np.histogram(contour_lengths, bins=7, range=(1, max(contour_lengths) + 1))
    return edge_histogram

def calculate_edge_angles(img):
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Calculate gradients in x and y directions
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate edge angles
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    angle[angle < 0] += 180  # Convert angles to range [0, 180]

    # Create a histogram of edge angles
    angle_histogram, _ = np.histogram(angle[edges > 0], bins=7, range=(0, 180))

    return angle_histogram

def calculate_aspect_ratio(contour):
    _, _, w, h = cv2.boundingRect(contour)
    return w / h if h > 0 else 0

def calculate_area_by_perimeter(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return area / perimeter if perimeter > 0 else 0

def calculate_hue_histogram(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_histogram, _ = np.histogram(hsv_img[:, :, 0], bins=7, range=(0, 180))
    return hue_histogram


def process_image(data):
    index, image_path = data
    if index % 100 == 0:
        print(f"Processing image at index: {index}")
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Keypoints and descriptors
    keypoints, descriptors = calculate_keypoints_and_descriptors(gray)
    n_keypoints = len(keypoints) if keypoints is not None else 0
    
    # Calculate average and RMS brightness from grayscale image
    avg_brightness = calculate_perceived_brightness(gray, 'avg')
    brightness_rms = calculate_perceived_brightness(gray, 'rms')

    # Calculate average and RMS of perceived brightness from RGB image
    perc_brightness_avg = calculate_perceived_brightness(image, 'avg')
    perc_brightness_rms = calculate_perceived_brightness(image, 'rms')

    # Calculate contrast
    contrast = calculate_contrast(gray)

    # Calculate edge length histogram
    edges = calculate_edges_new(image)
    edge_len_features = {f'edge_length{i+1}': edges[i] for i in range(len(edges))}
    # Calculate edge angle histogram
    edge_angles = calculate_edge_angles(image)
    edge_angle_features = {f'edge_angle{i+1}': edge_angles[i] for i in range(len(edge_angles))}

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None
    aspect_ratio = calculate_aspect_ratio(main_contour) if main_contour is not None else 0
    area_by_perim = calculate_area_by_perimeter(main_contour) if main_contour is not None else 0

    # Calculate hue histogram
    hue_histogram = calculate_hue_histogram(image)
    hue_features = {f'hue{i+1}': hue_histogram[i] for i in range(len(hue_histogram))}

    return {
        'filename': os.path.basename(image_path),
        'n_keypoints': n_keypoints,
        'avg_brightness': avg_brightness,
        'brightness_rms': brightness_rms,
        'avg_perc_brightness': perc_brightness_avg,
        'perc_brightness_rms': perc_brightness_rms,
        'contrast': contrast,
        **edge_len_features,
        **edge_angle_features,
        'area_by_perim': area_by_perim,
        'aspect_ratio': aspect_ratio,
         **hue_features
    }


# Replace with the actual directory of the ILSVRC 2012 dataset
dataset_dir = '../../../images/ILSVRC/Data/CLS-LOC/val/'

# Assuming images are directly under the dataset directory
image_paths = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)]
# image_paths = image_paths[:1000]  # Example with a subset of 1000 images

# Parallel processing
from concurrent.futures import ProcessPoolExecutor

# Pair each image path with its index
image_paths_with_index = list(enumerate(image_paths))

def process_image_parallel(image_data, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, image_data))
    return results

# Extract features for each image in parallel
features_list = process_image_parallel(image_paths_with_index, max_workers=8)  # Adjust max_workers as needed

# Convert to DataFrame and save to CSV
df = pd.DataFrame(features_list)
df.to_csv('ilsvrc2012_features.csv', index=False)

