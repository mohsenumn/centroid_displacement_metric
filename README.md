# Centroid Displacement Metric (CDM)

**Centroid Displacement Metric (CDM)** is a topological and geometric evaluation method specifically designed for assessing the performance of image segmentation techniques. This metric calculates the displacement of the centroids of the loops (or closed regions) present in both the original binary mask (ground truth) and the predicted binary mask. CDM is particularly useful for applications where topological accuracy is crucial, such as medical imaging, remote sensing, and Geographic Information Systems (GIS).

## Requirements

To use this repository, ensure you have the following libraries installed:
- `scipy`
- `opencv-python`
- `numpy`
- `shapely`
- `scikit-image`
- `matplotlib`
- `pandas`

You can install the required libraries with the following command:

```bash
pip install scipy opencv-python numpy shapely scikit-image matplotlib pandas
```

## Installation
To use this repository, clone it and install the required dependencies.

```bash
git clone https://github.com/your-username/CentroidDisplacementMetric.git
cd CentroidDisplacementMetric
pip install -r requirements.txt
```

## Usage

You can use the following code to calculate the Centroid Displacement Metric (CDM) for a given dataset path:

```python
p = "/path/to/directory/of/binary/masks"

results_df = get_CDM(p)
cdm = results_df['CDM'].mean()

print('CDM:  ', cdm)
```

### Parameters:
- `p`: Path to the folder containing the ground truth and predicted segmentations assuming ground truth masks start with `gt_` and predicted masks start with `pred_` and both end with `.png`. 

