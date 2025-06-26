# Cell Marker Spot Counter

This repository provides a Python tool for automated analysis of microscopy images to count green and red fluorescent spots per cell and detect their overlap. It is designed for multi-channel images, where each cell may contain red and/or green markers.


## Features

- **Counts** the number of green and red fluorescent spots per image.
- **Detects and counts** overlapping spots (co-localization) between green and red markers.
- **Batch processes** all `.tif` images in a directory.
- **Outputs** per-image results and a summary CSV file.
- **Saves binary masks** for each channel and overlap regions.
- **Adjustable thresholds** and overlap parameters.

## Usage
```
python cell_marker_spot_counter.py -i <input_dir> -o <output_dir> [options]
```

### Command Line

*Arguments:**
- `-i <input_dir>`: Path to directory containing `.tif` images or a single image file (**required**)
- `-o <output_dir>`: Path to directory for results (**required**)
- `-min_overlap_size <int>`: Minimal pixel area to count as overlap (default: 5)
- `-green_threshold <float>`: Minimum intensity for green marker (0–1, optional)
- `-red_threshold <float>`: Minimum intensity for red marker (0–1, optional)


## How It Works

1. **Channel Extraction:**  
   The tool loads each `.tif` image and extracts the red and green channels.

2. **Cell Segmentation:**  
   Adaptive thresholding and morphological operations identify and segment marker spots in each channel.

3. **Overlap Detection:**  
   Binary masks for green and red spots are compared to find overlapping regions, indicating co-localization.

4. **Result Output:**  
   - Annotated images and masks for each channel and overlap.
   - A summary CSV with counts for green spots, red spots, and overlaps per image.

## Output Files

For each image, the following are saved in the output directory:
- `green_cell_mask.png`: Binary mask of green spots.
- `red_cell_mask.png`: Binary mask of red spots.
- `cell_w_both_marker.png`: Binary mask of overlapping spots.
- `image_without_white.png`: Image with white/gray background suppressed.
- `summary.csv`: Table with per-image counts of green, red, and overlapping spots.

## Example Output Table

| Filename                      | Green spots | Red spots | Mixed spots |
|-------------------------------|-------------|-----------|-------------|
| VID1607_C1_2_00d00h00m.tif    | 12          | 7         | 3           |

## Requirements

- Python 3.7+
- numpy
- matplotlib
- scikit-image
- opencv-python
- scipy

