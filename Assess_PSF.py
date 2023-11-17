## Import Packages
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
from org.apache.commons.math3.fitting import PolynomialCurveFitter, WeightedObservedPoints
from java.lang import Math
from org.apache.commons.math3.linear import RealMatrix, EigenDecomposition
from ij import ImagePlus
from ij.process import ImageProcessor


# # # # # # # # # # # # # # # # # # # # SETTINGS # # # # # # # # # # # # # # # # # # # #

settings = {
    "base-folder":      "/home/shaswati/Documents/PSF/63x-confocal-ok2",
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

# Function to calculate intensity distribution of the filtered images
def calculate_intensity_distribution(imIn):
    # Get the image processor
    image_processor = imIn.getProcessor()
    
    # Get the pixel values
    intensity_values = list(image_processor.getPixels())
    
    return intensity_values

# Actual implementation to calculate the covariance matrix
def calculate_actual_covariance_matrix(intensity_values):
    # Convert intensity_values to a list
    intensity_values_list = list(intensity_values.getPixels())

    # Calculate the mean intensity
    mean_intensity = sum(intensity_values_list) / len(intensity_values_list)

    # Compute the deviations from the mean
    deviations = [x - mean_intensity for x in intensity_values_list]

    # Calculate the covariance matrix
    covariance_matrix = [[sum(deviations[i] * deviations[j] for i in range(len(deviations))) / len(deviations)
                          for j in range(len(deviations))]
                         for _ in range(len(deviations))]

    return covariance_matrix


    
# Function to calculate the moments of the PSF image
def calculate_psf_moments(imIn):
    # Convert the PSF image to an ImageProcessor
    psf_processor = imIn.getProcessor()
    
    # Calculate the second-order moments (covariance matrix) of the PSF image
    covariance_matrix = calculate_covariance_matrix(psf_processor)
    
    # Calculate the eigenvectors and eigenvalues of the covariance matrix
    eigenvectors, eigenvalues = calculate_eigenvectors_eigenvalues(covariance_matrix)
    
    # Determine the angle of bending with respect to a particular axis using the orientation represented by the eigenvectors
    bending_angle = determine_bending_angle(eigenvectors)
    
    return bending_angle

# Function to calculate the covariance matrix of the PSF image
def calculate_covariance_matrix(intensity_values):
    covariance_matrix = calculate_actual_covariance_matrix(intensity_values)
    return covariance_matrix

    

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

            # Calculate the moments of the PSF image
            psf_moments = calculate_psf_moments(labels)

            # Create a dictionary to store the labeled images and their corresponding bending angles
            data = {}
            for i in range(len(labels)):
                data[labels[i]] = psf_moments[i]

            # Convert the dictionary to a JSON object
            json_data = json.dumps(data)

            # Save the JSON object to a file
            with open('output.json', 'w') as f:
                f.write(json_data)

        # Close all open images (temporary)
        IJ.run("Close All")

# Call the main function to start processing the images
main()

     



