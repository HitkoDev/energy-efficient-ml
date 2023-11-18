import cv2
import numpy as np
import pandas as pd
import os
from skimage import color, feature, measure, filters

def calculate_brightness(img):
    return np.mean(img)

def calculate_rms(img):
    return np.sqrt(np.mean(np.square(img)))

def calculate_contrast(img):
    return img.std()

def calculate_edges(img):
    edges = feature.canny(color.rgb2gray(img))
    edge_lengths = measure.regionprops(measure.label(edges), intensity_image=img)[0].area
    edge_histogram, _ = np.histogram(edge_lengths, bins=7, range=(1, edge_lengths+1))
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

def process_image(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Feature extraction
    keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    n_keypoints = keypoints.shape[0] if keypoints is not None else 0

    avg_brightness = calculate_brightness(gray)
    brightness_rms = calculate_rms(gray)

    perceived_brightness = calculate_brightness(color.rgb2gray(image))
    perc_brightness_rms = calculate_rms(color.rgb2gray(image))

    contrast = calculate_contrast(gray)

    edges = calculate_edges(image)
    edge_angles = calculate_edge_angles(image)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None
    aspect_ratio = calculate_aspect_ratio(main_contour) if main_contour is not None else 0
    area_by_perim = calculate_area_by_perimeter(main_contour) if main_contour is not None else 0

    hue_histogram = calculate_hue_histogram(image)

    return {
        'filename': os.path.basename(image_path),
        'n_keypoints': n_keypoints,
        'avg_brightness': avg_brightness,
        'brightness_rms': brightness_rms,
        'avg_perc_brightness': perceived_brightness,
        'perc_brightness_rms': perc_brightness_rms,
        'contrast': contrast,
        'edge_length': edges.tolist(),  # Convert to list for CSV
        'edge_angles': edge_angles.tolist(),  # Convert to list for CSV
        'aspect_ratio': aspect_ratio,
        'area_by_perim': area_by_perim,
        'hue': hue_histogram.tolist()  # Convert to list for CSV
    }

# Replace with the actual directory of the ILSVRC 2012 dataset
dataset_dir = '../../../images/ILSVRC/Data/CLS-LOC/val/'

# Assuming images are directly under the dataset directory
image_paths = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)]
# print(image_paths[:10])
image_paths = image_paths[:1000]

# Extract features for each image
features_list = [process_image(path) for path in image_paths if path.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

# Convert to DataFrame and save to CSV
df = pd.DataFrame(features_list)
df.to_csv('ilsvrc2012_features.csv', index=False)
