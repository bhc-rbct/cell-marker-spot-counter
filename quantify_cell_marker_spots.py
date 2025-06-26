
import argparse
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imsave
import numpy as np
from numpy import ndarray
from skimage import measure, morphology
from skimage.filters import threshold_local, threshold_otsu, gaussian
from skimage.color import label2rgb
import skimage
import cv2
from skimage.measure import label, regionprops
from scipy import ndimage

def get_color_mask_props(
    used_channel: np.ndarray,
    white_mask,
    channel_name,
    output_path: str = None,
    set_threshold = None
) -> tuple:
   
    # Create output directory if specified
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    used_channel[white_mask] = used_channel[white_mask] * 0.1  

    gaussian_image = gaussian(used_channel, sigma=1, channel_axis=-1)

    background_mask = gaussian_image < threshold_otsu(gaussian_image)
    foreground_mask = ~background_mask

    gaussian_image[background_mask] = gaussian_image[background_mask] * 0.2  

    threshold = threshold_otsu(gaussian_image[foreground_mask]) if not set_threshold else set_threshold
    thresholded_image = (gaussian_image >= threshold).astype(bool)

    removed_holes_image = morphology.remove_small_holes(
        thresholded_image, connectivity=1, area_threshold=2) 

    removed_small_obj_image = morphology.remove_small_objects(removed_holes_image, min_size=10)

    # Detect connected components (cells)
    cell_labels = measure.label(removed_small_obj_image, connectivity=1)

    # Get properties of the labeled regions
    cell_props = measure.regionprops(cell_labels)

    # Create binary mask of cells
    binary_mask = (removed_small_obj_image > 0).astype(np.uint8) * 255  # Now 0 or 255

    # Save and display results
    if output_path:
        # output_file = os.path.join(output_path, f'{channel_name}_removed_holes_image.png')
        # cv2.imwrite(output_file, (removed_holes_image * 255).astype(np.uint8))
        mask_filename = os.path.join(output_path, f'{channel_name}_cell_mask.png')
        cv2.imwrite(mask_filename, binary_mask)

    return len(cell_props), binary_mask

def find_overlapping_spots(mask1, mask2, output_path, min_area=5):
    """
    Find overlapping spots between two binary masks.
    
    Parameters:
    - mask1: First binary mask (uint8 array with values 0 or 255)
    - mask2: Second binary mask (uint8 array with values 0 or 255)
    - min_area: Minimum area (in pixels) to consider as a valid spot
    
    Returns:
    - overlap_mask: Binary mask showing only overlapping regions
    - spot_count: Number of distinct overlapping spots
    """
    # Ensure masks are binary (0 or 255)
    mask1 = (mask1 > 0).astype(np.uint8) * 255
    mask2 = (mask2 > 0).astype(np.uint8) * 255
    
    # Find overlapping regions
    overlap = np.logical_and(mask1, mask2).astype(np.uint8) * 255
    
    # Label connected components in the overlap
    labeled, num_features = ndimage.label(overlap)
    
    # Filter small spots (optional)
    if min_area > 1:
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_area:
                overlap[labeled == i] = 0
        # Relabel after filtering
        labeled, num_features = ndimage.label(overlap)

    if output_path:
        output_file = os.path.join(output_path, 'cell_w_both_marker.png')
        cv2.imwrite(output_file, overlap)


    return num_features

def main(args):
    if os.path.isdir(args.input):
        input_dir_path = Path(args.input)
        output_path = args.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        all_data = []
        
        for file in os.listdir(input_dir_path):
            if file.endswith('.tif'):
                print(f"Running for {file}")
                # Create full input path
                file_path = os.path.join(input_dir_path, file)
                
                try:
                    image = skimage.io.imread(file_path)
                    file_base_name = os.path.splitext(file)[0]
                    
                    # Create per-image output directory
                    image_output_dir = os.path.join(output_path, file_base_name)
                    os.makedirs(image_output_dir, exist_ok=True)
                    
                    red_channel = image[:, :, 0]
                    green_channel = image[:, :, 1]

                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    is_white_gray = (hsv[:, :, 1] < 35)  # Low saturation

                    
                    green_spots, green_mask = get_color_mask_props(
                        green_channel,
                        is_white_gray,
                        "green",
                        image_output_dir,
                        args.green_threshold
                    )

                    red_spots, red_mask = get_color_mask_props(
                        red_channel,
                        is_white_gray,
                        "red",
                        image_output_dir,
                        args.red_threshold
                    )

                    both_signals = find_overlapping_spots(
                        green_mask, 
                        red_mask, 
                        image_output_dir, 
                        args.min_overlap_size
                    )

                    modified_image = skimage.io.imread(file_path)                    
                    modified_image[is_white_gray] = modified_image[is_white_gray] * 0.1

                    image_bgr = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)

                    output_file = os.path.join(image_output_dir, 'image_without_white.png')
                    cv2.imwrite(output_file, image_bgr)
                    
                    all_data.append([file, green_spots, red_spots, both_signals])
                
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue

        # Write summary CSV
        summary_path = os.path.join(output_path, 'summary.csv')
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Green spots', 'Red spots', 'Mixed spots'])
            writer.writerows(all_data)

        print(f"Processed {len(all_data)} images. Results saved to {summary_path}")

if __name__ == "__main__":
    # Argument Parsing
    description = (
        "Counts the number of red dots per cell (DAPI/blue signal)."
        "Can be called with a directory or individual image file. "
        "To exclude cells below a certain size, try increasing min_cell_size. If the cells contain holes try increasing -blue_hole_threshold. If cells need to be more seperated try decreasing -blue_hole_threshold. "
        "If too many/less red dots are identified increase/lower -red_bg_threshold. You can also try lowering -red_max_size "
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        help="Path to the directory which contains all .tif files, which should be analyzed (should not contain any other .tif files) or to the input file",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        help="Path to directory where the count.csv and annotated .tifs (annotated cells + marked red dots, which where counted; default: input_dir)",
        required=True,
    )

    parser.add_argument(
        "-min_overlap_size",
        dest="min_overlap_size",
        default=5,
        type=int,
        help="Minimal number of pixels to count the overlap of green and red (default: 5)",
        required=False,
    )

    parser.add_argument(
        "-green_threshold",
        dest="green_threshold",
        type=float,
        help="Minmal intensity of red to count as marker green signal (min: 0, max: 1) if not given calculated using threshold_otsu",
        required=False,
    )

    parser.add_argument(
        "-red_threshold",
        dest="red_threshold",
        type=float,
        help="Minmal intensity of red to count as marker red signal (min: 0, max: 1) if not given calculated using threshold_otsu",
        required=False,
    )

    args = parser.parse_args()
    main(args)
