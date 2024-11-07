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

def clean_all_from_junk(directory):
    def delete_junk(f):
        try:
            shutil.rmtree(f)
        except:
            os.unlink(f)
    msk_dirs = os.listdir(directory)  
    for f in msk_dirs:
        if f.startswith('.') or f == '__pycache__' or f.endswith('txt'):
            junk_p = os.path.join(directory, f)
            print('deleted-->', junk_p)
            delete_junk(junk_p)
        else:
            if os.path.isdir(f):
                subfolder_p = os.path.join(directory, f)
                clean_all_from_junk(subfolder_p)
    print('All cleaned!')

def split_string(filename):
    match = re.match(r"([^_]+_)(.+)", filename)
    if match:
        return [match.group(1), match.group(2)]
    else:
        return [filename] 

def get_CDM(p):
    clean_all_from_junk(p)
    errors = []
    for pt in os.listdir(p):
        nn = split_string(pt)[1]
        mask_gt = imread(f'{p}/gt_{nn}')[:, :, 1]
        if mask_gt.max() == 1:
            mask_gt = (255 * mask_gt).astype('uint8')
        mask_pred = imread(f'{p}/pred_{nn}')
        if mask_pred.max() == 1:
            mask_pred = (255 * mask_pred).astype('uint8')
        CDM_norm, CDM = structure_similarity(mask_gt, mask_pred, plot=False)
        errors.append([CDM_norm, CDM, pt])
    df = pd.DataFrame(errors, columns=['CDM_norm', 'CDM', 'im_name'])
    return df




