import csv
import os
import math
import json

from ij import IJ, ImagePlus, ImageStack
from ij.plugin.filter import BackgroundSubtracter
from ij.process import AutoThresholder, StackStatistics
from ij.measure import ResultsTable
from ij.gui import Plot

from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling3D
from inra.ijpb.label import LabelImages
from inra.ijpb.plugins import AnalyzeRegions3D

from imagescience.feature import Laplacian
from imagescience.image import Image
from imagescience.image import FloatImage


# # # # # # # # # # # # # # # # # # # # SETTINGS # # # # # # # # # # # # # # # # # # # #

settings = {
    "base-folder":      "/home/benedetti/Documents/projects/PSF/63x-confocal-bad",
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


def subtract_background(imIn):
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


def psf_to_labels(imIn, title):
    # Preparing folder for control images.
    exportDir = os.path.join(settings["base-folder"], settings["dir-masks"])
    if not os.path.isdir(exportDir):
        os.mkdir(exportDir)

    # Saving calibration for later use
    calib = imIn.getCalibration()

    # Applying LoG filter.
    laplacian = Laplacian()
    image = Image.wrap(imIn)
    output = FloatImage(image)
    output = laplacian.run(output, settings["LoG-radius"])
    res = output.imageplus()

    # Convert to mask
    stack = res.getStack()
    threshold_method = AutoThresholder.Method.Otsu
    thresholder = AutoThresholder()

    out = ImageStack(stack.getWidth(), stack.getHeight())

    stack_stats = StackStatistics(res)
    long_histogram = stack_stats.getHistogram()
    histogram = [int(value) for value in long_histogram]
    thresholdBin = thresholder.getThreshold(threshold_method, histogram)

    hMin = stack_stats.histMin
    hMax = stack_stats.histMax
    threshold = hMin + ((hMax - hMin) / stack_stats.nBins) * thresholdBin
    IJ.log("Thresholding at " + str(threshold))

    for i in range(1, res.getStackSize() + 1):
        ip = stack.getProcessor(i)
        ip.setThreshold(-1e30, threshold)
        nip = ip.createMask()
        out.addSlice(nip)

    res.close()

    # Labeling by connected components
    ffrcl = FloodFillRegionComponentsLabeling3D(26, 16)
    labeled_stack = ffrcl.computeLabels(out, 255)  # 26-connectivity and 16-bit image
    
    LabelImages.removeBorderLabels(labeled_stack)
    labels_list = [l for l in LabelImages.findAllLabels(labeled_stack) if l > 0]
    IJ.log(str(len(labels_list)) + " PSFs found on the image.")

    clean_labels = ImagePlus("mask_"+title, labeled_stack)
    clean_labels.setCalibration(calib)

    exportPath = os.path.join(exportDir, "mask_"+title+".tif")
    IJ.saveAs(clean_labels, "Tiff", exportPath)

    return clean_labels


def get_calibrated_dimensions(imIn):
    calib = imIn.getCalibration()
    
    width = imIn.getWidth()
    height = imIn.getHeight()
    depth = imIn.getNSlices()

    width *= calib.pixelWidth
    height *= calib.pixelHeight
    depth *= calib.pixelDepth

    return (width, height, depth)


def filter_psfs(labels, title):
    # Preparing folder for CSVs.
    exportDirData   = os.path.join(settings["base-folder"], settings["dir-data"])
    exportDirLabels = os.path.join(settings["base-folder"], settings["dir-labels"])

    if not os.path.isdir(exportDirLabels):
        os.mkdir(exportDirLabels)
    
    if not os.path.isdir(exportDirData):
        os.mkdir(exportDirData)

    # Extracting properties of labels
    analyze_regions = AnalyzeRegions3D()
    rsl = analyze_regions.process(labels)
    
    exportPathRawData = os.path.join(exportDirData, "raw_"+title+".csv")
    rsl.saveAs(exportPathRawData)

    headings = rsl.getHeadings()

    # Checking that required data is available

    if (not _cx in headings) or (not _cy in headings) or (not _cz in headings):
        IJ.log("[!!!] Centroids are required but not available.")
        return (labels, None)
    
    if (not _bb_min_x in headings) or (not _bb_min_y in headings) or (not _bb_min_z in headings):
        IJ.log("[!!!] Bounding-boxes are required but not available.")
        return (labels, None)

    if (not _bb_max_x in headings) or (not _bb_max_y in headings) or (not _bb_max_z in headings):
        IJ.log("[!!!] Bounding-boxes are required but not available.")
        return (labels, None)

    # Filtering PSFs according to tolerable distances from borders
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

        # Discard the PSF if it is too close from another one
        for i in range(rsl.size()):
            if i == current_row:
                continue
            (x2, y2, z2) = (rsl.getValue(_cx, i), rsl.getValue(_cy, i), rsl.getValue(_cz, i))

            # If the distance is the same as the threshold, continue
            dist = ((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) ** 0.5
            if dist <= settings['dist-psf']:
                IJ.log("PSF [" + str(current_row) + "] discarded due to its proximity with [" + str(i) + "] (" + str(dist) + ") um.")
                continue

        good_lbls.add(current_row+1)
        
        clean_results.addRow()
        
        clean_results.addValue(_lbl, current_row+1)

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

    exportPathData   = os.path.join(exportDirData, "locations_"+title+".csv")
    exportPathLabels = os.path.join(exportDirLabels, "labels_"+title+".tif")

    IJ.saveAs(clean_labels, "Tiff", exportPathLabels)
    clean_results.saveAs(exportPathData)

    return (clean_labels, clean_results)


def check_swap(p1, p2):
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
    angle = math.radians(settings["ang-step"])
    calib = imIn.getCalibration()
    plots = {}

    for current_row in range(locations.size()):
        label = int(locations.getValue(_lbl, current_row))
        (x, y, z) = (locations.getValue(_cx, current_row), locations.getValue(_cy, current_row), locations.getValue(_cz, current_row))
        len_x = locations.getValue(_bb_max_x, current_row) - locations.getValue(_bb_min_x, current_row)
        len_y = locations.getValue(_bb_max_y, current_row) - locations.getValue(_bb_min_y, current_row)
        len_z = locations.getValue(_bb_max_z, current_row) - locations.getValue(_bb_min_z, current_row)
        plane_xy = max(len_x, len_y)
        plane_z = len_z
        radius = plane_xy/2
        rad_h = plane_z/2
        sums = []

        for i in range(int(settings["max-angle"]/settings["ang-step"])):
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

            (p1, p2) = check_swap(p1, p2)

            # Loop through each voxel in the PSF image and calculate the sum of all voxels intersecting the plane
            accumulator = 0
            stack = imIn.getStack()

            x_start = int(calib.getRawX(p1[0]))
            y_start = int(calib.getRawY(p1[1]))
            z_start = int(calib.getRawZ(p1[2]))

            x_end = int(calib.getRawX(p2[0]))
            y_end = int(calib.getRawY(p2[1]))
            z_end = int(calib.getRawZ(p2[2]))

            for z_index in range(z_start, z_end+1):
                for y_index in range(y_start, y_end+1):
                    for x_index in range(x_start, x_end+1):
                        accumulator += stack.getVoxel(x_index, y_index, z_index)

                
            # Add the sum to the list of sums
            sums.append(accumulator)
            angles = [a for a in range(0, settings["max-angle"], settings["ang-step"])]

        plots[label] = sums
    
    return plots


def save_plots_to_file(plots, title):
    exportDir = os.path.join(settings["base-folder"], "plots")
    if not os.path.isdir(exportDir):
        os.mkdir(exportDir)

    exportPath = os.path.join(exportDir, "radial_profiles_" + title + ".json")
    json_object = json.dumps(plots, indent=4) 
    with open(exportPath, 'wb') as f:
        f.write(json_object)



def locate_psfs(imIn):
    """
    Function taking an image (stack) representing a PSF and producing a labels map, containing the bounding volumes of each PSF.
    
    Args:
        imIn (ImagePlus): An image on black background representing PSFs on a field.
    
    Returns:
        (ImagePlus) A labeled image with a label == a PSF.
    """
    title = imIn.getTitle().lower().replace(" ", "_").split(".")[0] # Building clean name
    subtract_background(imIn) # Subtracting the irregular background
    labels = psf_to_labels(imIn, title) # Labeling PSFs

    return (labels, title)


def main():
    # Process each 3D TIFF image in the folder
    content = [c for c in os.listdir(settings['base-folder']) if os.path.isfile(os.path.join(settings['base-folder'], c))]
    for k, file_name in enumerate(content):
        try:
            full_path = os.path.join(settings['base-folder'], file_name)
            imIn = IJ.openImage(full_path)
        except:
            pass
        else:
            IJ.log("\n=========== Processing: " + file_name + " [" + str(k+1) + "/" + str(len(content)) + "] ===========")
            labels, base_title = locate_psfs(imIn)
            labels, locations = filter_psfs(labels, base_title)
            profiles = radial_profiling(imIn, locations)
            save_plots_to_file(profiles, base_title)

        IJ.run("Close All") # temporary
        return


main()
