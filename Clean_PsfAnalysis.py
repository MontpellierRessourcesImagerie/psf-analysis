import csv
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
from threading import Thread
from ij.plugin import Duplicator
from ij.process import ImageProcessor
from inra.ijpb.plugins import AnalyzeRegions
from imagescience.transform import Translate
from imagescience.transform import Rotate
from imagescience.transform import Transform
from imagescience.image import Axes
from java.util import ArrayList
# # # # # # # # # # # # # # # # # # # # SETTINGS # # # # # # # # # # # # # # # # # # # #

settings = {
    "base-folder":      "/home/shaswati/Documents/PSF/60x-1.42_old-banana",
    "threshold-method": "Otsu",
    "dist-psf":         1.5, # Tolerable distance (in um) between two PSFs, or from a PSF to a border.
    "ball-radius":      50,
    "LoG-radius":       0.2,
    "dir-labels":       "labels",
    "dir-masks":        "masks",
    "dir-data":         "locations",
    "max-angle":        180,
    "ang-step":         2,
    "crop-x":           16,
    "crop-y":           16
}

_lbl = "Label"

_cx = "Centroid.X"
_cy = "Centroid.Y"
_cz = "Centroid.Z"

_center_x ="Centeroid.x"
_center_y ="Centeroid.y"

_bb_min_x = "Box.X.Min"
_bb_min_y = "Box.Y.Min"
_bb_min_z = "Box.Z.Min"

_bb_max_x = "Box.X.Max"
_bb_max_y = "Box.Y.Max"
_bb_max_z = "Box.Z.Max"
_b_angles = "Elli.Roll"
_sorted_elli_roll = "Sorted Elli Roll"
_b_azim  = 'Elli.Azim'

####################################################################

# Create a dialog to allow the user to set settings
gd = GenericDialog("PSF Processing Settings")
gd.addStringField("Base Folder:", settings["base-folder"])
gd.addChoice("Threshold Method:", ["Otsu", "AnotherMethod"], settings["threshold-method"])
gd.addNumericField("Tolerable Distance (um):", settings["dist-psf"], 2)
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

###########################################################################

def subtract_background(imIn):
    """
    Subtract background from the input image.
    
    @param imIn: Input image to subtract background from.
    @type imIn : ImagePlus
       
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
    """ 
    Normalize the pixel values of an image to the range [0.0, 1.0]. 
    
    @param imIn: Input image to normalize. 
    @type imIn: ImagePlus
    
    """
    
    # Get the image's statistics, including min and max pixel values
    stats = ImageStatistics.getStatistics(imIn.getProcessor())
    
    # Calculate the range of the pixel values
    pixel_range = stats.max - stats.min
    
    # Normalize each slice of the image
    for i in range(1, imIn.getNSlices() + 1):
        imIn.setSlice(i)
        ip = imIn.getProcessor()
            
        for x in range(ip.getWidth()):
            for y in range(ip.getHeight()):
                value = ip.getPixelValue(x, y)
                if pixel_range != 0:
                  normalized_value = (value - stats.min) / pixel_range
                else:
                    normalized_value = 0  
                ip.putPixelValue(x, y, normalized_value)

def psf_to_labels(imIn, title):
    """
    Convert the input PSF image to labeled regions.
    
    @param imIn: Input image containing PSFs.
    @type imIn : ImagePlus
    @param title: Title of the image.
    @type title: str
    
    @return: Labeled image containing PSF regions.
    @rtype: ImagePlus
    
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

    out = ImageStack(res.getWidth(), res.getHeight())
    
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
    
    @param imIn: Input image
    @type imIn: ImagePlus
    
    @return:Calibrated width, height, and depth.
    @rtype:tuple
    
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

    @param labels:Labeled image containing PSF regions.
    @type labels: ImagePlus
    @param title:  Title of the image
    @title type: str
    
    @return: Cleaned labels and filtered results.
    @rtype: tuple
        
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

    if(not _b_angles in headings):
        IJ.log ("[!!!] Bending-angles are required but not available")
        return (labels, None)
    
    if(not _b_azim in headings):
        IJ.log ("[!!!] Azimute-angles are required but not available")
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
        b_ang = rsl.getValue(_b_angles, current_row)
        b_az = rsl.getValue(_b_azim, current_row)

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

        clean_results.addValue(_b_angles,b_ang)
        clean_results.addValue(_b_azim,b_az)
        
        elli_azim =rsl.getValue(_b_azim, current_row)
        elli_roll = rsl.getValue(_b_angles, current_row)
        if (80 <= elli_roll <= 120) or (-80 >= elli_roll >=-120):
            sorted_elli_roll = 1
        else:
            sorted_elli_roll = -1
        clean_results.addValue(_sorted_elli_roll, sorted_elli_roll)
        

    IJ.log(str(clean_results.size()) + " left after filtering.")
    clean_labels = LabelImages.keepLabels(labels, [i for i in good_lbls])
    labels.close()

    exportPathData = os.path.join(exportDirData, "locations_" + title + ".csv")
    exportPathLabels = os.path.join(exportDirLabels, "labels_" + title + ".tif")

    IJ.saveAs(clean_labels, "Tiff", exportPathLabels)
    clean_results.saveAs(exportPathData)

    return (clean_labels, clean_results)

def distance_3d(p1, p2):
    """
    Calculate 3D-distance between two points
    
    @param p1 = one point in the format (x,y,z)
    @type p1 = tuple
    @param p2 = second point in format (x,y,z)
    @type p2 = tuple
    
    @return: The 3D distance between the two types
    @rtype: tuple
    
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def create_blank_canvas(title, width, height, depth):
    
    """ 
    Create a blank canvas with the specified title, width, height, and depth
    
    @param title: The title of the canvas
    @type title: str
    @param width: The width of the canvas
    @param height: The height of the canvas
    @param depth: The depth of the canvas
    
    @return: The created blank canvas
    @rtype: Image
    """
    result_img = IJ.createImage("HeatMap-"+title, "32-bit black", width, height, depth)
    result_img.getProcessor().add(0.5) 
    result_img.show()
    print (width,height,depth)
    return result_img

def unpack_centeroids(results_table,calib):
    centeroids = []
    for i in range(results_table.size()):
        x = results_table.getValue(_cx,i)
        y = results_table.getValue(_cy,i)
        z = results_table.getValue(_cz,i)
        # Apply calibration to adjust the dimensions
        x/=  calib.pixelWidth
        y/= calib.pixelHeight
        z/= calib.pixelDepth    
        centeroids.append((int(x),int(y),int(z)))
    print(centeroids)
    return centeroids   
    

def get_centroid_coordinates(centroids):
    """
    Get the coordinates of the centroids
    
    @param centroids: The list of centroids
    @type centroids: List[Tuple]
    @return: The coordinates of the centroids
    """
    coordinates = []
    for centroid in centroids:
        x, y, z = centroid
        coordinates.append((x, y, z))
    return coordinates

def crop_around_psf(clean_labels, locations, imIn, settings):
    """
    Crop an image around each PSF using centroids and perform image transformations
    
    @param clean_labels: The image to be cropped
    @type clean_labels: ImagePlus
    @param locations: The list of centroid locations
    @type locations: List[Tuple[int, int, int]]
    @param imIn: The input image
    @type imIn: ImagePlus
    @param settings: The settings for cropping
    @type settings: dict
    @param elli_roll: The roll angle for rotation
    @type elli_roll: float
    @param elli_azim: The azimuth angle for rotation
    @type elli_azim: float
    @return: The list of cropped images
    @rtype: List[ImagePlus]
    """
    
    # Define the translation values
    tx = settings["crop-x"] / 2
    ty = settings["crop-y"] / 2
    
    # Perform image transformations and cropping for each centroid
    cropped_images = []
    calibration = imIn
    centroids = unpack_centeroids(locations, calibration)
    duplicator = Duplicator()
    for centroid in centroids:
        x, y, z = centroid
        startX = int(x - (settings["crop-x"] / 2))
        startY = int(y - (settings["crop-y"] / 2))
        
        # Set the ROI using Image.setRoi
        clean_labels.setRoi(startX, startY,settings["crop-x"],settings["crop-y"])
        
        # Create a Duplicator and crop the image   
        cropped_image = duplicator.crop(clean_labels)
                
        # Perform the image transformations
        transform = Transform()
        transform.translate(tx, Axes.X)
        transform.translate(ty, Axes.Y)
        translation1 = transform.getMatrix()
        transform.transform(translation1)
        
        rotation = transform.rotate(elli_roll, elli_azim)
        transform.transform(rotation)
        
        # Create the second translation matrix to move the image back to its original position
        translation2 = transform.getMatrix()
        transform.translate(-tx, Axes.X)
        transform.translate(-ty, Axes.Y)
        transform.transform(translation2)
        
        # Append the transformed image to the list
        cropped_images.add(cropped_image)
        
    return cropped_images

   
    #Calculate Centeroids per slice in the Cropped Image
    def calculate_slice_centroids(im):
        centroids_per_slice = []
        for z in range(1, im.getNSlices() + 1):
            im.seSlice(z) 
            ar = AnalyzeRegions()
            # Analyze Regions
            regions = ar.process(im)
            
            # Find the region with the largest area
            biggest_area = 0
            index = 0
            for i in range(regions.number_of_objects()):
                area = regions.getValue("Area", i)
                if area > biggest_area:
                    biggest_area = area
                    index = i
        
        # Get the centroid of the region with the largest area
        center_x = regions.getValue(_cx, index)
        center_y = regions.getValue(_cy, index)
        
        centroids_per_slice.append((center_x, center_y, z))
    
    return centroids_per_slice

    # Calculate the distance from the first centroid of the first slice
    
    def distance_profile(centroids_per_slice):
        reference_centroid = centroids_per_slice[0]      
        for i in range(1, len(centroids_per_slice)+1):
            reference_centroids[2]= i
            distance.append(distance(reference_centeroid, centroids_per_slice[i]))
    return distance

def save_plots_to_file(distance, title):
    """
    Save centeroids of each slice of cropped image to a JSON file.

    @param plots: A dictionary with Centeroids as keys and distance between them as values
    @param title: The title used for the output JSON file.
    @type title: str
    """
    exportDir = os.path.join(settings["base-folder"], "plots")
    
    # Create the output directory if it doesn't exist
    if not os.path.isdir(exportDir):
        os.mkdir(exportDir)

    # Define the path for the output JSON file
    exportPath = os.path.join(exportDir, "centeroids_profiles_" + title + ".json")

    # Serialize the plots to JSON format with indentation
    json_object = json.dumps(plots, indent=4) 

    # Write the JSON object to the output file
    with open(exportPath, 'wb') as f:
        f.write(json_object)

      
class ProcessRegionThread(Thread):


    def __init__(self, region, centroid_list, properties, result_img,i):
        Thread.__init__(self)
        self.region = region
        self.centroid_list = centroid_list
        self.properties = properties
        self.result_img = result_img
        

    def run(self):
        x_start, y_start, z_start = self.region[0]
        x_end, y_end, z_end = self.region[1]
        for i in range(x_start, x_end):
            for h in range(y_start, y_end):
                for s in range(z_start, z_end):
                    accumulator = 0
                    for k, centroid in enumerate(self.centroid_list):
                        d = distance_3d((i, h, s), centroid)
                        score = self.properties.getValue("Sorted Elli Roll", k)

                        if d < 0.0001:
                            accumulator = score
                            break

                        weight = 1 / d
                        accumulator += (score * weight)
                    self.result_img.getStack().setVoxel(i, h, s, accumulator)
                    


def weighted_average_3d(properties, title, width, height, depth, calib, n_threads):
    """
    Calculate the weighted average in 3D using multithreading
    
    @param properties: The properties object
    @param title: The title of the canvas
    @type title: str
    @param width: The width of the canvas
    @type width: Number
    @param height: The height of the canvas
    @type height: Number
    @param depth: The depth of the canvas
    @type depth: Number
    @param calib: The calibration object
    @type calib: object
    @param n_threads: The number of threads to use
    
    """
    
    result_img = create_blank_canvas(title, width, height, depth)
    centroid_list = unpack_centeroids(properties, calib)

    # Calculate the number of slices per thread
    slices_per_thread = depth // n_threads

    threads = []

    # Create and start threads
    for i in range(n_threads):
        z_start = i * slices_per_thread
        z_end = (i+1) * slices_per_thread
        #z_end = depth if i == n_threads - 1 else (i + 1) * slices_per_thread
        region = ((0, 0, z_start), (width, height, z_end))
        thread = ProcessRegionThread(region, centroid_list, properties, result_img,i)
        thread.start()
        threads.append(thread)
        
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

def check_swap(p1, p2):
    """
    Ensure that p1 and p2 define a consistent bounding box.

    @param p1: The first point defining the bounding box.
    @type p1: tuple 
    @param p2: The second point defining the bounding box.
    @type p2: tuple
    @return: Two points defining a consistent bounding box.
    @rtype: tuple
    
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

    @param labeled_stacks: Labeled image stack.
    @type labeled_stacks: ImagePlus
    @return: Dilated labeled image stack.
    @rtype: ImagePlus
    
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

    @param imIn: An image representing PSFs on a black background.
    @type imIn: ImagePlus
    @return: A labeled image with labeled PSFs and a clean title.
    @rtype: ImagePlus, str 
    
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

def main():
    # Get a list of 3D TIFF images in the specified folder
    content = [c for c in os.listdir(settings['base-folder']) if os.path.isfile(os.path.join(settings['base-folder'], c))]

    # Iterate through each image in the folder
    for k, file_name in enumerate(content):
        try:
            full_path = os.path.join(settings['base-folder'], file_name)
            imIn = IJ.openImage(full_path)
        except:
            pass
        else:
            # Log a message indicating the start of processing for the current image
            IJ.log("\n=========== Processing: " + file_name + " [" + str(k+1) + "/" + str(len(content)) + "] ===========")
            
            # Locate PSFs and obtain labeled image and title
            labels, base_title = locate_psfs(imIn)

            # Filter PSFs and get filtered labels and locations
            labels, locations = filter_psfs(labels, base_title)
            
            #Crop the image, gather the co-ordinates of the centeroids, transform the matrix and calculate the intensity slice by slice 
            #Calculate the centeroids distance and obtain profile 
            cropped_versions = crop_around_psf(labels, locations,imIn, settings)
            distances = []
            for cropped in cropped_versions:
                distances.append(calculate_slice_centroids(cropped))
                cropped.close()
            # Save profiling plots to a JSON file
            save_plots_to_file(distances, base_title)
            
            #Calculate weighted average and generate a heatmap
            _
            width, height, depth = imIn.getWidth(), imIn.getHeight(), imIn.getNSlices()

            result_image = weighted_average_3d(locations,imIn.getTitle(),width, height, depth,imIn.getCalibration(),10)
            
    return result_image

# Close all open images (temporary)
IJ.run("Close All")

# Call the main function to start processing the images
main()





