from scipy.spatial import KDTree
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon
from skimage.draw import line as draw_line
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import dilation
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import os, shutil, re
import pandas as pd

import os
import numpy as np
from skimage.io import imread
from scipy.ndimage import label, binary_fill_holes
import matplotlib.pyplot as plt

def measure_betti_numbers(file_path, plot=False):
    structure = np.ones((3, 3), dtype=int)
    # delta_betti0 = []
    # delta_betti1 = []
    # for file_name in os.listdir(directory_path):
    #     file_path = os.path.join(directory_path, file_name)
    #     if os.path.isfile(file_path) and file_name.endswith(".png") and file_name.startswith("gt_"):
            # Ground Truth
    im_g = imread(file_path, as_gray=True) 
    if im_g.max() > 1:
        im_g = im_g/255
    gt = (dilation(im_g)).astype('uint8')
    labeled_array_gt, betti0_gt = label(gt, structure=structure)
    filled_array_gt = binary_fill_holes(gt)
    holes_array_gt = filled_array_gt.astype(int) - gt.astype(int)
    labeled_holes_gt, betti1_gt = label(holes_array_gt, structure=structure)

    # Prediction
    pred_file_path = file_path.replace('gt_', 'pred_')
    im_p = imread(pred_file_path, as_gray=True) 
    if im_p.max() > 1:
        im_p = im_p/255
    p = (dilation(im_p)).astype('uint8')
    labeled_array_p, betti0_p = label(p, structure=structure)
    filled_array_p = binary_fill_holes(p)
    holes_array_p = filled_array_p.astype(int) - p.astype(int)
    labeled_holes_p, betti1_p = label(holes_array_p, structure=structure)

    # CDM_norm, CDM = structure_similarity(mask_gt, mask_pred, plot=False)
    # accuracy, precision, recall, dice, iou = compute_metrics_from_images(mask_gt/255, mask_pred/255)
    # errors.append([CDM, accuracy, precision, recall, dice, iou])

    delta_betti0= betti0_gt - betti0_p
    delta_betti1= betti1_gt - betti1_p
    
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        # Ground Truth Visualization
        axes[0, 0].imshow(gt, cmap='gray')
        axes[0, 0].set_title(f"Ground Truth (GT)\nBetti0: {betti0_gt}, Betti1: {betti1_gt}")
        axes[0, 1].imshow(labeled_array_gt, cmap='nipy_spectral')
        axes[0, 1].set_title(f"Connected Components (GT)")

        axes[1, 0].imshow(holes_array_gt, cmap='gray')
        axes[1, 0].set_title("Holes Array (GT)")
        axes[1, 1].imshow(labeled_holes_gt, cmap='nipy_spectral')
        axes[1, 1].set_title(f"Holes (GT)\nBetti1: {betti1_gt}")

        # Prediction Visualization
        axes[2, 0].imshow(p, cmap='gray')
        axes[2, 0].set_title(f"Prediction (P)\nBetti0: {betti0_p}, Betti1: {betti1_p}")
        axes[2, 1].imshow(labeled_array_p, cmap='nipy_spectral')
        axes[2, 1].set_title(f"Connected Components (P)")

        plt.tight_layout()
        plt.show()
    return delta_betti0, delta_betti1
    # return round(np.array(delta_betti0).mean(),2), round(np.array(delta_betti0).std(),2), round(np.array(delta_betti1).mean(),2), round(np.array(delta_betti1).std(),2)


def polygonize(mask, erosion=0):
    if erosion > 0:
        mask = binary_erosion(mask, iterations=erosion).astype('uint8')
    elif erosion < 0:
        mask = binary_dilation(mask, iterations=(-erosion)).astype('uint8')
    binary_image = mask.copy()
    binary_image = np.pad(binary_image, 2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    threshold_area = 10
    polygons = []
    bps = []
    for label in range(1, num_labels):
        roi = (labels == label).astype(np.uint8)
        contour, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.contourArea(contour[0]) < threshold_area:
            continue
        boundary_coordinates = np.squeeze(contour[0]).tolist()
        if len(boundary_coordinates) > 2:
            bps.append(boundary_coordinates)
            polygon = Polygon(boundary_coordinates)
            polygons.append(polygon.convex_hull)
    return polygons

def rasterize(inp_obj, array_size=None, centroids=True):
    def reverse_padding(array, padding):
        return array[padding:-padding, padding:-padding]

    def draw_circle(center, r, array, px_value=2):
        array_size = array.shape
        y, x = np.ogrid[:array_size[0], :array_size[1]]
        cir = (x - center[0])**2 + (y - center[1])**2 <= r**2
        array[cir] = px_value
        return array

    if array_size is None:
        array = None
    else:
        array = np.zeros(array_size)
        array = np.pad(array, 1)

    def rasterize_single_polygon(obj, array):
        nonlocal array_size
        if obj.geom_type == 'Polygon':
            xy = np.array(obj.exterior.coords)
            rr, cc = draw_polygon(xy[:, 1], xy[:, 0])
            if array is None:
                max1 = max([np.max(el) for el in xy]).astype('int') + 2
                array_size = (max1, max1)
                array = np.zeros(array_size, dtype=np.uint8)
            array[rr, cc] = 1
        else:
            print('type error!')
        return array

    cneties = []
    if (type(inp_obj) == list) or (type(inp_obj) == tuple):
        if array is None:
            xy_g = [np.array(obj.exterior.coords) for obj in inp_obj]
            max_dim = max([np.max(el) for el in xy_g]).astype('int') + 2
            array_size = (max_dim, max_dim)
            array = np.zeros(array_size, dtype=np.uint8)
        for obj in inp_obj:
            array = rasterize_single_polygon(obj, array=array)
            if centroids:
                cneties.append(list(obj.centroid.coords)[0])
                array = draw_circle(list(obj.centroid.coords)[0], 1, array, px_value=0.5)
    else:
        array = rasterize_single_polygon(inp_obj, array=array)
        if centroids:
            array = draw_circle(list(inp_obj.centroid.coords)[0], 1, array, px_value=2)
    array = reverse_padding(array, 1)
    return (array * 255).astype('uint8'), cneties

def nearest_distances(centroids_list_gt, centroids_list_pred):
    tree = KDTree(centroids_list_pred)
    distances, _ = tree.query(centroids_list_gt)
    normalized_mean_distance = distances.mean() / distances.max()
    return (1 - normalized_mean_distance) * 100, distances.mean()

def structure_similarity(mask_gt1, mask_pred1, erosion_times=2, plot=False):
    mask_pred = 255 - mask_pred1
    mask_pred_ = np.where(mask_pred > mask_pred.mean(), 255, 0).astype('uint8')
    mask_pred_polys = polygonize(mask_pred_, erosion=erosion_times)
    mask_pred_with_cents, centroids_list_pred = rasterize(mask_pred_polys, array_size=None, centroids=True)
    
    mask_gt = 255 - mask_gt1
    mask_gt_ = np.where(mask_gt > mask_gt.mean(), 255, 0).astype('uint8')
    mask_gt_polys = polygonize(mask_gt_, erosion=erosion_times)
    mask_gt_with_cents, centroids_list_gt = rasterize(mask_gt_polys, array_size=None, centroids=True)

    if mask_gt_with_cents.shape != mask_pred_with_cents.shape:
        target_shape = (max(mask_gt_with_cents.shape[0], mask_pred_with_cents.shape[0]),
                        max(mask_gt_with_cents.shape[1], mask_pred_with_cents.shape[1]))
        mask_gt_with_cents = resize(mask_gt_with_cents, target_shape, preserve_range=True, anti_aliasing=False).astype('uint8')
        mask_pred_with_cents = resize(mask_pred_with_cents, target_shape, preserve_range=True, anti_aliasing=False).astype('uint8')

    CDM_norm, CDM = nearest_distances(centroids_list_gt, centroids_list_pred)
    CDM_norm = round(CDM_norm, 4)
    CDM = round(CDM, 4)

    if plot:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.dstack((mask_gt_, mask_pred, np.zeros_like(mask_gt))).astype('uint8'))
        plt.title('Masks Overlay', fontsize=20)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.dstack((mask_gt_with_cents, mask_pred_with_cents, np.zeros_like(mask_pred_with_cents))).astype('uint8'))
        plt.axis('off')
        plt.show()
    return CDM_norm, CDM


def split_string(filename):
    match = re.match(r"([^_]+_)(.+)", filename)
    if match:
        return [match.group(1), match.group(2)]
    else:
        return [filename] 

def get_CDM_standards(p):
    errors = []
    for pt in os.listdir(p):
        nn = split_string(pt)[1]
        file_path = f'{p}/gt_{nn}'
        delta_betti0, delta_betti1 = measure_betti_numbers(file_path, plot=False)
        mask_gt = imread(file_path)[:, :, 1]
        mask_gt = ((mask_gt))
        if mask_gt.max() == 1:
            mask_gt = (255 * mask_gt).astype('uint8')
        mask_pred = imread(file_path.replace('gt', 'pred'))
        mask_pred = ((mask_pred))
        if mask_pred.max() == 1:
            mask_pred = (255 * mask_pred).astype('uint8')
        try:
            CDM_norm, CDM = structure_similarity(mask_gt, mask_pred, plot=False)
        except:
            CDM = 999
        accuracy, precision, recall, dice, iou = compute_metrics_from_images(mask_gt/255, mask_pred/255)
        
        errors.append([accuracy, precision, recall, dice, iou, delta_betti0, delta_betti1, CDM])
    # df = pd.DataFrame(errors, columns=['CDM', 'accuracy', 'precision', 'recall', 'dice', 'iou'])
    return errors

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import os
from PIL import Image
import numpy as np
import pandas as pd

def compute_metrics_from_images(gt, pred):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    tn = np.sum((1 - pred) * (1 - gt))
    fn = np.sum((1 - pred) * gt)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return round(100*accuracy, 2), round(100*precision, 2), round(100*recall, 2), round(100*dice, 2), round(100*iou, 2)


import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def process_directory_parallel(fol, path):
    directory_path = os.path.join(path, fol)
    errors = get_CDM_standards(directory_path)  # Replace with your function
    df = pd.DataFrame(errors, columns=['accuracy', 'precision', 'recall', 'dice', 'iou', 'delta_betti0', 'delta_betti1', 'CDM'])

    summary_dict = {}
    for col in df.columns:
        summary_dict[f"{col}_avg"] = [df[col].mean()]
        summary_dict[f"{col}_std"] = [df[col].std()]
    
    summary_df = pd.DataFrame(summary_dict)
    summary_df['exp'] = fol
    summary_df = summary_df[['exp'] + [col for col in summary_df.columns if col != 'exp']]
    return summary_df


betti_error_dict = {}
# pp = "dendro_unet"
pp = "dendro_topo"
path = f"OUTPUT/DENDRO/{pp}/"
clean_all_from_junk("OUTPUT")  
drs = os.listdir(path)
columns_tot = ['exp','accuracy_avg','accuracy_std','precision_avg','precision_std','recall_avg','recall_std','dice_avg','dice_std','iou_avg','iou_std','delta_betti0_avg','delta_betti0_std','delta_betti1_avg','delta_betti1_std','CDM_avg','CDM_std']

final_df = pd.DataFrame(columns=columns_tot)

with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda fol: process_directory_parallel(fol, path), drs))

final_df = pd.concat(results, ignore_index=True)
final_df.round(2).to_csv(f'TOTAL_{pp}.csv', index=False)
