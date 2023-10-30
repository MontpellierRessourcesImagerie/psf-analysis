import csv
import os
from ij import IJ
from ij.measure import ResultsTable
from ij.gui import WaitForUserDialog
from ij import ImagePlus
from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling3D
from inra.ijpb.label import LabelImages
from inra.ijpb.plugins import AnalyzeRegions3D


#create a dictionary with all my settings
settings = {
    "base-folder": "/home/shaswati/Documents/PSF/40x-1.4-banana",
    "threshold-method": "Default",
    "border-trd-xy": 10, # Define the threshold in pixels for x, y, and z axes
    "border-trd-z": 5,
    "dist-psf": 10
}

def locate_psfs(full_path):

    # Open the 3D image
    image = IJ.openImage(full_path)

    IJ.run(image, "Subtract Background...", "rolling=50 stack")
    image.show()
    rsl_good = image.getTitle()

    lapresult = IJ.getImage()

    mask = lapresult.duplicate()
    lapresult.close()

    # ImageConverter.setDoScaling(True)
    IJ.setAutoThreshold(mask, "Default dark stack")
    IJ.run(mask, "Convert to Mask", "Background=dark")
    mask.getProcessor().invert()
    
    # Use FloodFillRegionComponentsLabeling3D (FFRC) to label the connected components
    ffrcl = FloodFillRegionComponentsLabeling3D(26, 16)
    labeled_image = ffrcl.computeLabels(mask.getStack(), 255)  # 26-connectivity and 16-bit image
    
    # Use LabelImages to remove border regions
    LabelImages.removeBorderLabels(labeled_image)

    # Analyze regions using AnalyzeRegions3D
    clean_image = ImagePlus("PSF_rsl", labeled_image)
    IJ.saveAs(clean_image, "Tiff", os.path.join(settings['base-folder'], rsl_good + "_mask.tif"))

    analyze_regions = AnalyzeRegions3D()
    rsl = analyze_regions.process(clean_image)
    
    data_path = os.path.join(settings['base-folder'], rsl_good + ".csv")
    rsl.saveAs(data_path)
    return (clean_image, data_path)



def filter_psfs(clean_image, data_path):

    # Load the CSV table of PSF centroids
    table_path = data_path
    table = []

    with open(table_path, "r") as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            table.append(row)

    # Loop through each PSF in the table
    psf_clean= []
    for i in range(0, len(table)):
        # Get the centroid coordinates for this PSF
        x = float(table[i][13])
        y = float(table[i][14])
        z = float(table[i][15])

        # Calculate the distance from the borders
        x_dist = min(x, clean_image.getWidth() - x)
        y_dist = min(y, clean_image.getHeight() - y)
        z_dist = min(z, clean_image.getStackSize() - z)

        # Discard PSFs that are too close to the borders
        if x_dist < settings['border-trd-xy'] or y_dist < settings['border-trd-xy'] or z_dist < settings['border-trd-z']:
            continue


        # Loop through the table again to find neighboring PSFs
        for j in range(0, len(table)):
            if i == j:
                continue
            # Get the centroid coordinates for this neighboring PSF
            x2 = float(table[j][13])
            y2 = float(table[j][14])
            z2 = float(table[j][15])

            # Calculate the distance between the two PSFs
            dist = ((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2) ** 0.5

            # If the distance is the same as the threshold, continue
            if dist < settings['dist-psf']:
                continue

        # Add the PSF and its neighbors to the list of PSFs
        psf_clean.append((x,y,z))

    # Write the PSFs to a CSV file
    with open(data_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","z"])
        for psf in psf_clean:
            writer.writerow(psf)
    
    clean_image.close()
      
            
def main():
    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(settings['base-folder']):
        full_path = os.path.join(settings['base-folder'], file_name)
        
        if not file_name.lower().endswith(".tif"):
            continue

        if "mask" in file_name:
            continue

        clean_image, data_path = locate_psfs(full_path)
        filter_psfs(clean_image, data_path)

        IJ.run("Close All")


main()

