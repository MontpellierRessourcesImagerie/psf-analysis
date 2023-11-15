import json
import os
import math
from ij import IJ, ImagePlus, ImageStack
from ij.plugin.filter import BackgroundSubtracter
from ij.process import AutoThresholder, StackStatistics
from ij.measure import ResultsTable
from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling3D
from inra.ijpb.label import LabelImages
from inra.ijpb.plugins import AnalyzeRegions3D
from imagescience.feature import Laplacian
from imagescience.image import Image
from imagescience.image import FloatImage
from ij.gui import GenericDialog
from ij.process import ImageStatistics

# # # # # # # # # # # # # # # # # # # # SETTINGS # # # # # # # # # # # # # # # # # # # #

settings = {
    "base-folder":      "/home/shaswati/Documents/PSF/63x-confocal-ok",
    "threshold-method": "Otsu",
    "dist-psf":         1.5, # Tolerable distance (in µm) between two PSFs, or from a PSF to a border.
    "ball-radius":      50,
    "LoG-radius":       0.2,
    "dir-labels":       "labels",
    "dir-masks":        "masks",
    "dir-data":         "locations",
    "max-angle":        180,
    "ang-step":         2
}

_lbl = "Label"

_cx = "Centroid.X"
_cy = "Centroid.Y"
_cz = "Centroid.Z"

_bb_min_x = "Box.X.Min"
_bb_min_y = "Box.Y.Min"
_bb_min_z = "Box.Z.Min"

_bb_max_x = "Box.X.Max"
_bb_max_y = "Box.Y.Max"
_bb_max_z = "Box.Z.Max"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Create a dialog to allow the user to set settings
gd = GenericDialog("PSF Processing Settings")
gd.addStringField("Base Folder:", settings["base-folder"])
gd.addChoice("Threshold Method:", ["Otsu", "AnotherMethod"], settings["threshold-method"])
gd.addNumericField("Tolerable Distance (µm):", settings["dist-psf"], 2)
gd.addNumericField("Ball Radius:", settings["ball-radius"], 0)
gd.addNumericField("LoG Radius:", settings["LoG-radius"], 1)
gd.addStringField("Directory for Labels:", settings["dir-labels"])
gd.addStringField("Directory for Masks:", settings["dir-masks"])
gd.addStringField("Directory for Data:", settings["dir-data"])
gd.addNumericField("Max Angle (degrees):", settings["max-angle"], 0)
gd.addNumericField("Angle Step (degrees):", settings["ang-step"], 0)
gd.showDialog()

# Check if the user canceled the dialog
if gd.wasCanceled():
    IJ.log("User canceled the dialog. Using default settings.")
else:
    # Retrieve user-set values from the dialog
    settings["base-folder"] = gd.getNextString()
    settings["threshold-method"] = gd.getNextChoice()
    settings["dist-psf"] = gd.getNextNumber()
    settings["ball-radius"] = int(gd.getNextNumber())
    settings["LoG-radius"] = gd.getNextNumber()
    settings["dir-labels"] = gd.getNextString()
    settings["dir-masks"] = gd.getNextString()
    settings["dir-data"] = gd.getNextString()
    settings["max-angle"] = int(gd.getNextNumber())
    settings["ang-step"] = int(gd.getNextNumber())

def subtract_background(imIn):
    """
    Subtract background from the input image.

    Args:
        imIn (ImagePlus): Input image to subtract background from.
    """
    b = BackgroundSubtracter()

    for n in range(1, imIn.getNSlices()+1):
        imIn.setSlice(n)
        b.rollingBallBackground(
            imIn.getProcessor(),
            settings["ball-radius"],
            False,
            False,
            False,
            False,
            True
        )

def normalize_image(imIn):
    """ Normalize the pixel values of an image to the range [0.0, 1.0].
    
    Args:
        imIn (ImagePlus): Input image to normalize.
    """
    # Get the image's statistics, including min and max pixel values
    stats = ImageStatistics.getStatistics(imIn.getProcessor())
    
    # Calculate the range of the pixel values
    pixel_range = stats.max - stats.min
    
    # Normalize each slice of the image
    for i in range(1, imIn.getNSlices() + 1):
        imIn.setSlice(i)
        ip = imIn.getProcessor()
        
        # Adjust each pixel value
        for x in range(ip.getWidth()):
            for y in range(ip.getHeight()):
                value = ip.getPixelValue(x, y)
                normalized_value = (value - stats.min) / pixel_range
                ip.putPixelValue(x, y, normalized_value)



def psf_to_labels(imIn, title):
    """
    Convert the input PSF image to labeled regions.

    Args:
        imIn (ImagePlus): Input image containing PSFs.
        title (str): Title of the image.

    Returns:
        ImagePlus: Labeled image containing PSF regions.
    """
    # Create a folder for control images
    exportDir = os.path.join(settings["base-folder"], settings["dir-masks"])
    if not os.path.isdir(exportDir):
        os.mkdir(exportDir)

    # Save calibration for later use
    calib = imIn.getCalibration()

    # Apply Laplacian of Gaussian (LoG) filter
    laplacian = Laplacian()
    image = Image.wrap(imIn)
    output = FloatImage(image)
    output = laplacian.run(output, settings["LoG-radius"])
    res = output.imageplus()

    # Convert the filtered image to a mask
    stack = res.getStack()
    threshold_method = AutoThresholder.Method.Otsu
    thresholder = AutoThresholder()

    out = ImageStack(stack.getWidth(), stack.getHeight())

    stack_stats = StackStatistics(res)
    long_histogram = stack_stats.getHistogram()
    histogram = [int(value) for value in long_histogram]
    threshold_bin = thresholder.getThreshold(threshold_method, histogram)

    hMin = stack_stats.histMin
    hMax = stack_stats.histMax
    threshold = hMin + ((hMax - hMin) / stack_stats.nBins) * threshold_bin
    IJ.log("Thresholding at " + str(threshold))

    for i in range(1, res.getStackSize() + 1):
        ip = stack.getProcessor(i)
        ip.setThreshold(-1e30, threshold)
        nip = ip.createMask()
        out.addSlice(nip)

    res.close()

    # Label the regions using connected components
    ffrcl = FloodFillRegionComponentsLabeling3D(26, 16)
    labeled_stack = ffrcl.computeLabels(out, 255)  # 26-connectivity and 16-bit image

    # Remove border labels
    LabelImages.removeBorderLabels(labeled_stack)
    labels_list = [l for l in LabelImages.findAllLabels(labeled_stack) if l > 0]
    IJ.log(str(len(labels_list)) + " PSFs found on the image.")

    clean_labels = ImagePlus("mask_" + title, labeled_stack)
    clean_labels.setCalibration(calib)

    exportPath = os.path.join(exportDir, "mask_" + title + ".tif")
    IJ.saveAs(clean_labels, "Tiff", exportPath)

    return clean_labels

def get_calibrated_dimensions(imIn):
    """
    Get the dimensions of the input image after applying calibration.

    Args:
        imIn (ImagePlus): Input image.

    Returns:
        tuple: Calibrated width, height, and depth.
    """
    calib = imIn.getCalibration()
    
    # Get the original dimensions
    width = imIn.getWidth()
    height = imIn.getHeight()
    depth = imIn.getNSlices()

    # Apply calibration to adjust the dimensions
    width *= calib.pixelWidth
    height *= calib.pixelHeight
    depth *= calib.pixelDepth

    return (width, height, depth)

def filter_psfs(labels, title):
    """
    Filter PSFs based on various criteria and save the filtered results.

    Args:
        labels (ImagePlus): Labeled image containing PSF regions.
        title (str): Title of the image.

    Returns:
        tuple: Cleaned labels and filtered results.
    """
    # Create folders for CSVs
    exportDirData = os.path.join(settings["base-folder"], settings["dir-data"])
    exportDirLabels = os.path.join(settings["base-folder"], settings["dir-labels"])

    if not os.path.isdir(exportDirLabels):
        os.mkdir(exportDirLabels)
    
    if not os.path.isdir(exportDirData):
        os.mkdir(exportDirData)

    # Extract properties of labels
    analyze_regions = AnalyzeRegions3D()
    rsl = analyze_regions.process(labels)
    
    exportPathRawData = os.path.join(exportDirData, "raw_" + title + ".csv")
    rsl.saveAs(exportPathRawData)

    headings = rsl.getHeadings()

    # Check if required data is available
    if (not _cx in headings) or (not _cy in headings) or (not _cz in headings):
        IJ.log("[!!!] Centroids are required but not available.")
        return (labels, None)
    
    if (not _bb_min_x in headings) or (not _bb_min_y in headings) or (not _bb_min_z in headings):
        IJ.log("[!!!] Bounding-boxes are required but not available.")
        return (labels, None)

    if (not _bb_max_x in headings) or (not _bb_max_y in headings) or (not _bb_max_z in headings):
        IJ.log("[!!!] Bounding-boxes are required but not available.")
        return (labels, None)

    # Filter PSFs according to tolerable distances from borders
    (width, height, depth) = get_calibrated_dimensions(labels)
    clean_results = ResultsTable()
    good_lbls = set()

    for current_row in range(rsl.size()):
        (x, y, z) = (rsl.getValue(_cx, current_row), rsl.getValue(_cy, current_row), rsl.getValue(_cz, current_row))
        (x_dist, y_dist, z_dist) = (min(x, width - x), min(y, height - y), min(z, depth - z))

        # Discard PSFs that are too close to the borders
        if min(x_dist, y_dist, z_dist) < settings['dist-psf']:
            IJ.log("PSF [" + str(current_row) + "] discarded due to its proximity with the border (" + str(min(x_dist, y_dist, z_dist)) + ") um.")
            continue

        # Discard the PSF if it is too close to another one
        for i in range(rsl.size()):
            if i == current_row:
                continue
            (x2, y2, z2) = (rsl.getValue(_cx, i), rsl.getValue(_cy, i), rsl.getValue(_cz, i))

            # If the distance is the same as the threshold, continue
            dist = ((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) ** 0.5
            if dist <= settings['dist-psf']:
                IJ.log("PSF [" + str(current_row) + "] discarded due to its proximity with [" + str(i) + "] (" + str(dist) + ") um.")
                continue

        good_lbls.add(current_row + 1)
        
        clean_results.addRow()
        
        clean_results.addValue(_lbl, current_row + 1)

        clean_results.addValue(_cx, x)
        clean_results.addValue(_cy, y)
        clean_results.addValue(_cz, z)

        clean_results.addValue(_bb_min_x, rsl.getValue(_bb_min_x, current_row))
        clean_results.addValue(_bb_min_y, rsl.getValue(_bb_min_y, current_row))
        clean_results.addValue(_bb_min_z, rsl.getValue(_bb_min_z, current_row))

        clean_results.addValue(_bb_max_x, rsl.getValue(_bb_max_x, current_row))
        clean_results.addValue(_bb_max_y, rsl.getValue(_bb_max_y, current_row))
        clean_results.addValue(_bb_max_z, rsl.getValue(_bb_max_z, current_row))

    IJ.log(str(clean_results.size()) + " left after filtering.")
    clean_labels = LabelImages.keepLabels(labels, [i for i in good_lbls])
    labels.close()

    exportPathData = os.path.join(exportDirData, "locations_" + title + ".csv")
    exportPathLabels = os.path.join(exportDirLabels, "labels_" + title + ".tif")

    IJ.saveAs(clean_labels, "Tiff", exportPathLabels)
    clean_results.saveAs(exportPathData)

    return (clean_labels, clean_results)

def check_swap(p1, p2):
    """
    Ensure that p1 and p2 define a consistent bounding box.

    Args:
        p1 (tuple): The first point defining the bounding box.
        p2 (tuple): The second point defining the bounding box.

    Returns:
        tuple: Two points defining a consistent bounding box.
    """
    # Create a consistent bounding box by finding the minimum and maximum coordinates
    pa = (
        min(p1[0], p2[0]),
        min(p1[1], p2[1]),
        min(p1[2], p2[2])
    )

    pb = (
        max(p1[0], p2[0]),
        max(p1[1], p2[1]),
        max(p1[2], p2[2])
    )

    return (pa, pb)

def radial_profiling(imIn, locations):
    """
    Perform radial profiling of 
    Args:
        imIn (ImagePlus): Input image containing PSFs.
        locations (ResultsTable): Table containing PSF locations and properties.

    Returns:
        dict: A dictionary with PSF labels as keys and radial profiles as values.
    """
    angle = math.radians(settings["ang-step"])
    calib = imIn.getCalibration()
    plots = {}

    for current_row in range(locations.size()):
        # Extract information about the current PSF
        label = int(locations.getValue(_lbl, current_row))
        (x, y, z) = (locations.getValue(_cx, current_row), locations.getValue(_cy, current_row), locations.getValue(_cz, current_row))
        len_x = locations.getValue(_bb_max_x, current_row) - locations.getValue(_bb_min_x, current_row)
        len_y = locations.getValue(_bb_max_y, current_row) - locations.getValue(_bb_min_y, current_row)
        len_z = locations.getValue(_bb_max_z, current_row) - locations.getValue(_bb_min_z, current_row)
        plane_xy = max(len_x, len_y)
        plane_z = len_z
        radius = plane_xy / 2
        rad_h = plane_z / 2
        sums = []

        for i in range(int(settings["max-angle"] / settings["ang-step"])):
            # Rotate the plane for radial profiling
            rotate = i * angle

            p1 = (-radius, -radius, -rad_h)
            p2 = (radius, radius, rad_h)

            p1 = (
                p1[0] * math.cos(rotate) - math.sin(rotate) * p1[1],
                p1[0] * math.sin(rotate) + math.cos(rotate) * p1[1],
                -rad_h
            )

            p2 = (
                p2[0] * math.cos(rotate) - math.sin(rotate) * p2[1],
                p2[0] * math.sin(rotate) + math.cos(rotate) * p2[1],
                rad_h
            )

            p1 = (
                p1[0] + x,
                p1[1] + y,
                p1[2] + z
            )

            p2 = (
                p2[0] + x,
                p2[1] + y,
                p2[2] + z
            )

            # Ensure that p1 and p2 define a consistent bounding box
            (p1, p2) = check_swap(p1, p2)

            # Bounds checking to ensure indices are within stack dimensions
            x_start = int(calib.getRawX(p1[0]))
            y_start = int(calib.getRawY(p1[1]))
            z_start = int(calib.getRawZ(p1[2]))

            x_end = int(calib.getRawX(p2[0]))
            y_end = int(calib.getRawY(p2[1]))
            z_end = int(calib.getRawZ(p2[2]))

            x_start = max(0, x_start)
            y_start = max(0, y_start)
            z_start = max(0, z_start)

            x_end = min(imIn.getWidth() - 1, x_end)
            y_end = min(imIn.getHeight() - 1, y_end)
            z_end = min(imIn.getNSlices() - 1, z_end)

            # Loop through each voxel in the PSF image and calculate the sum of all voxels intersecting the plane
            accumulator = 0
            stack = imIn.getStack()

            for z_index in range(z_start, z_end + 1):
                for y_index in range(y_start, y_end + 1):
                    for x_index in range(x_start, x_end + 1):
                        accumulator += stack.getVoxel(x_index, y_index, z_index)

            # Add the sum to the list of sums
            sums.append(accumulator)

        # Store the radial profile for the current PSF
        plots[label] = sums

    return plots

            

def save_plots_to_file(plots, title):
    """
    Save radial profiling plots to a JSON file.

    Args:
        plots (dict): A dictionary with PSF labels as keys and radial profiles as values.
        title (str): The title used for the output JSON file.
    """
    exportDir = os.path.join(settings["base-folder"], "plots")
    
    # Create the output directory if it doesn't exist
    if not os.path.isdir(exportDir):
        os.mkdir(exportDir)

    # Define the path for the output JSON file
    exportPath = os.path.join(exportDir, "radial_profiles_" + title + ".json")

    # Serialize the plots to JSON format with indentation
    json_object = json.dumps(plots, indent=4) 

    # Write the JSON object to the output file
    with open(exportPath, 'wb') as f:
        f.write(json_object)

def dilate_labels(labeled_stack):
    """
    Apply dilation to the labeled regions using standard ImageJ functions.

    Args:
        labeled_stack (ImagePlus): Labeled image stack.

    Returns:
        ImagePlus: Dilated labeled image stack.
    """
    # Create a structuring element (3D ball) for dilation
    radius = settings["ball-radius"]
    stack = labeled_stack.getStack()
    width = stack.getWidth()
    height = stack.getHeight()
    n_slices = stack.getSize()
    se = ImageStack.create(width, height, n_slices, 32)  # 32 for 32-bit float data

    for z in range(n_slices):
        for y in range(height):
            for x in range(width):
                if (x - radius) ** 2 + (y - radius) ** 2 + (z - radius) ** 2 <= radius ** 2:
                    se.setVoxel(x, y, z, 255)

    # Apply dilation to the labeled image
    dilated_stack = labeled_stack.duplicate()

    for i in range(1, n_slices + 1):
        slice = stack.getProcessor(i)
        seImage = slice.duplicate()
        seImage.copyBits(se.getProcessor(i), 0, 0, 3)
        dilated_stack.getStack().setProcessor(seImage, i)

    return dilated_stack

def locate_psfs(imIn):
    """
    Locate and label point spread functions (PSFs) in an image stack.

    Args:
        imIn (ImagePlus): An image representing PSFs on a black background.

    Returns:
        (ImagePlus, str) A labeled image with labeled PSFs and a clean title.
    """
    # Generate a clean title for the output files
    title = imIn.getTitle().lower().replace(" ", "_").split(".")[0]

    # Subtract the irregular background from the input image
    subtract_background(imIn)

    # Normalize the image
    normalize_image(imIn)
    
    # Label PSFs in the image and get the labeled image
    labels = psf_to_labels(imIn, title)

    # Apply dilation to the labeled regions
    dilated_labels = dilate_labels(labels)

    # Return the labeled image and the clean title
    return dilated_labels, title

# Identify peaks
def find_peaks(imIn, threshold):
    peaks = []
    for i in range(1, len(imIn) - 1):
        if imIn[i] > threshold and imIn[i] > imIn[i - 1] and imIn[i] > [i + 1]:
            peaks.append(i)
    return peaks

def determine_bending_angles(peaks):
    angles = []
    for i in range(len(peaks) - 1):
        angle_rad = math.acos((peaks[i + 1] - peaks[i]) / (peaks[i + 1] + peaks[i]))
        angle_deg = math.degrees(angle_rad)
        angles.append(angle_deg)
    return angles


# Additional function to categorize angles
def categorize_angles_list(angles_list):
    categories_list = []

    for angles in angles_list:
        categories = {'0-30': 0, '31-60': 0, '61-90': 0, '91-120': 0, '121-150': 0, '151-180': 0}
        for angle in angles:
            if 0 <= angle < 30:
                categories['0-30'] += 1
            elif 30 <= angle < 60:
                categories['31-60'] += 1
            elif 60 <= angle < 90:
                categories['61-90'] += 1
            elif 90 <= angle < 120:
                categories['91-120'] += 1
            elif 120 <= angle < 150:
                categories['121-150'] += 1
            elif 150 <= angle <= 180:
                categories['151-180'] += 1
        categories_list.append(categories)

    return categories_list

def save_categories_to_file(plots, title):
    """
    Save radial profiling plots to a JSON file.

    Args:
        plots (dict): A dictionary with PSF labels as keys and radial profiles as values.
        title (str): The title used for the output JSON file.
    """
    exportDir = os.path.join(settings["base-folder"], "plots")

    # Define the path for the output JSON file
    exportPath = os.path.join(exportDir, "categories_profiles" + title + ".json")

    # Serialize the plots to JSON format with indentation
    json_object = json.dumps(plots, indent=4) 

    # Write the JSON object to the output file
    with open(exportPath, 'wb') as f:
        f.write(json_object)

# Main function
def main():
    # Get a list of 3D TIFF images in the specified folder
    content = [c for c in os.listdir(settings['base-folder']) if os.path.isfile(os.path.join(settings['base-folder'], c))]
    # Iterate through each image in the folder
    for k, file_name in enumerate(content):
        try:
            full_path = os.path.join(settings['base-folder'], file_name)
            imIn = IJ.openImage(full_path)
            if imIn is None:
                IJ.log("Failed to open the image: " + full_path)
                continue  # Skip to the next image if the current one cannot be opened
        except Exception as e:
            IJ.log("An error occurred while processing the image: " + str(e))
            continue  # Skip to the next image if an error occurs
        else:
            # Log a message indicating the start of processing for the current image
            IJ.log("\n=========== Processing: " + file_name + " [" + str(k + 1) + "/" + str(len(content)) + "] ===========")

            # Locate PSFs and obtain labeled image and title
            labels, base_title = locate_psfs(imIn)

            # Filter PSFs and get filtered labels and locations
            labels, locations = filter_psfs(labels, base_title)

            # Perform radial profiling and obtain profiles
            profiles = radial_profiling(imIn, locations)

            # Save radial profiling plots to a JSON file
            save_plots_to_file(profiles, base_title)

            # Normalize the image
            normalize_image(imIn)

            #Identify peaks

            # Extract bending angles from profiles
            threshold = 50000
            bending_angles_list = [determine_bending_angles(profile) for profile in profiles.values()]

            # Categorize bending angles
            categorized_angles_list = categorize_angles_list(bending_angles_list)

            # Save categorized angles as JSON
            save_categories_to_file(profiles, base_title)

        # Close all open images (temporary)
        IJ.run("Close All")

# Call the main function to start processing the images
main()






