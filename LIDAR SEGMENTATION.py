# Install the required packages in your repository's as follows
# pip install segment-lidar
# pip install cloth-simulation-filter

import os
import json
import pdal
from segment_lidar import samlidar

# Replace with the appropriate file paths in your GitHub repository
input_laz = "/path/to/Hotel_Southampton.laz"
output_las = "/path/to/Trees.las"

# Define the PDAL pipeline
pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": input_laz
        },
        {
            "type": "writers.las",
            "filename": output_las
        }
    ]
}

# Create a PDAL pipeline and execute it
pipeline = pdal.Pipeline(json.dumps(pipeline))
pipeline.execute()

# Initialize the SamLidar model with the correct checkpoint path
model = samlidar.SamLidar(ckpt_path="/path/to/sam_vit_h_4b8939.pth")

# Configure model parameters
model.crop_n_layers = 1
model.crop_n_points_downscale_factor = 2
model.min_mask_region_area = 500
model.points_per_side = 10
model.pred_iou_thresh = 0.90
model.stability_score_thresh = 0.92

# Replace with the appropriate LAS file path in your repository
las_file_path = "/path/to/Chemical_Plant.las"

# Read the LAS file
points = model.read(las_file_path)

# Perform ground segmentation
cloud, non_ground, ground = model.csf(points)

# Segment the points and provide image and labels paths
labels, *_ = model.segment(points=cloud, image_path="/path/to/raster.tif", labels_path="/path/to/labeled.tif")

# Write the segmented data to a LAS file
model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path="/path/to/segmented.las")
