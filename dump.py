
def part2():
    print("part2")
    # Import necessary Fiji classes
    from ij import IJ
    from ij.measure import ResultsTable
    from ij.plugin.frame import RoiManager
    from inra.ijpb.plugins import ConnectedComponentsLabeling
    import os

    # Set the path to your 3D TIFF images
    image_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana/40X_PSF_OIL_OLYMPUS_05.vsi - C488.tiff"
    output_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana/Output_PSF.txt"

    # Initialize a ResultsTable to store measurement results
    results_table = ResultsTable()

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".tif"):
            # Open the 3D image
            image = IJ.openImage(os.path.join(image_folder, file_name))

            # Threshold the 3D image
            IJ.setAutoThreshold(image, "Default dark")

            # Convert to binary using the threshold
            IJ.run(image, "Convert to Mask", "")
            
            # Run the MorphoLibJ plugin (Connected Components Labeling)
            IJ.run(image, "Connected Components Labeling", "connectivity=26 type=32-bits")
            
            # Measure properties of the regions
            IJ.run("Set Measurements...", "centroid area bounding area_fraction")
            IJ.run("Analyze Particles...", "display clear include summarize")

            # Get the results and add them to the ResultsTable
            current_results = ResultsTable.getResultsTable()
            results_table.incrementCounter()
            results_table.addValue("Image", file_name)
            results_table.addValue("Centroid_X", current_results.getValue("XM"))
            results_table.addValue("Centroid_Y", current_results.getValue("YM"))
            results_table.addValue("Centroid_Z", current_results.getValue("ZM"))
            results_table.addValue("Area", current_results.getValue("Area"))
            results_table.addValue("Bounding_Box", current_results.getValue("Bounding Area"))
            results_table.updateResults()

            # Add ROIs to the RoiManager
            roi_manager.addRoi(image.getRoi())

    # Save the ResultsTable and ROIs
    results_table.save(output_folder + "results.csv")
    roi_manager.runCommand("Save", output_folder + "rois.zip")

    # Close all images
    IJ.run("Close All")

    # Close the ResultsTable and RoiManager
    results_table.close()
    roi_manager.close()

def part1 ():
    # Import necessary Fiji classes
    from ij import IJ
    from ij.plugin.frame import RoiManager
    from inra.ijpb.plugins import ConnectedComponentsLabeling
    import os

    # Set the path to your TIFF images
    image_folder = "path/to/your/images/"
    output_folder = "path/to/output/folder/"

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each TIFF image in the folder
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".tif"):
            # Open the image
            image = IJ.openImage(os.path.join(image_folder, file_name))

            # Apply threshold (adjust the threshold level as needed)
            IJ.setAutoThreshold(image, "Default dark")
            IJ.run(image, "Convert to Mask", "")

            # Run the MorphoLibJ plugin for connected components labeling (adjust parameters as needed)
            plugin = IJ.runPlugIn("MorphoLibJ_.jar", "Analyze Label Image...", "")
            plugin.setConnectedComponentsLabeling(ConnectedComponentsLabeling.EIGHT_CONNECTED)
            plugin.setMinSize(100)  # Adjust the minimum size as needed

            # Add ROIs to the RoiManager
            roi_manager.addRoi(image.getRoi())

    # Save the ROIs
    roi_manager.runCommand("Save", output_folder + "rois.zip")

    # Close all images
    IJ.run("Close All")


def part3():
    # Import necessary Fiji classes
    from ij import IJ
    from ij.measure import ResultsTable
    from ij.plugin.frame import RoiManager
    from inra.ijpb.label import LabelImages
    from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling
    from ij.plugin.filter import BackgroundSubtracter
    import os

    # Set the path to your 3D TIFF images
    # Set the path to your 3D TIFF images directory
    image_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana"
    #image_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana/40X_PSF_OIL_OLYMPUS_05.vsi - C488.tiff"
    output_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana/Output_PSF.csv"


    # Initialize a ResultsTable to store measurement results
    results_table = ResultsTable()

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, file_name)
        if file_name.lower().endswith(".tif"):
            # Your processing code for 3D TIFF files goes here


            # Open the 3D image
            image = IJ.openImage(os.path.join(image_folder, file_name))

            # Threshold the 3D image
            IJ.setAutoThreshold(image, "Default dark")

            # Convert to binary using the threshold
            IJ.run(image, "Convert to Mask", "")
            
            # Apply background subtraction if needed
            bg_subt.rollingBallBackground(image, 50, False, False, True, True, False)
            
            # Label the connected components using FloodFillRegionComponentsLabeling
            labeled_image = FloodFillRegionComponentsLabeling.labelComponents(image, 8, 8, 0)
            
            # Measure properties of the regions
            IJ.run("Set Measurements...", "centroid area bounding area_fraction")
            IJ.run("Analyze Particles...", "display clear include summarize")

            # Get the results and add them to the ResultsTable
            current_results = ResultsTable.getResultsTable()
            results_table.incrementCounter()
            results_table.addValue("Image", file_name)
            results_table.addValue("Centroid_X", current_results.getValue("XM"))
            results_table.addValue("Centroid_Y", current_results.getValue("YM"))
            results_table.addValue("Centroid_Z", current_results.getValue("ZM"))
            results_table.addValue("Area", current_results.getValue("Area"))
            results_table.addValue("Bounding_Box", current_results.getValue("Bounding Area"))
            results_table.updateResults()

            # Add ROIs to the RoiManager
            roi_manager.addRoi(image.getRoi())

    # Save the ResultsTable and ROIs
    results_table.save(output_folder + "results.csv")
    roi_manager.runCommand("Save", output_folder + "rois.zip")

    # Close all images
    IJ.run("Close All")

    # Close the ResultsTable and RoiManager
    results_table.close()
    roi_manager.close()


def part5():
    # Import necessary Fiji classes
    from ij import IJ
    from ij.measure import ResultsTable
    from ij.plugin.frame import RoiManager
    from inra.ijpb.label import LabelImages
    from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling
    from ij.plugin.filter import BackgroundSubtracter
    import os

    # Set the path to your 3D TIFF images
    image_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana"
    output_folder = "C:/Users/ssarbagna/Documents/PSF/40x-1.4-banana/Output_PSF.csv"

    # Initialize a ResultsTable to store measurement results
    results_table = ResultsTable()

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, file_name)
        if file_name.lower().endswith(".tif"):
            # Open the 3D image
            image = IJ.openImage(os.path.join(image_folder, file_name))

            # Threshold the 3D image
            IJ.setAutoThreshold(image, "Default dark")

            # Convert to binary using the threshold
            IJ.run(image, "Convert to Mask", "")

            # Create an instance of BackgroundSubtracter
            bg_subt = BackgroundSubtracter()

            # Apply background subtraction
            bg_subt.rollingBallBackground(image, 50, False, False, True, True, False)

            # Label the connected components using FloodFillRegionComponentsLabeling
            labeled_image = FloodFillRegionComponentsLabeling.labelComponents(image, 8, 8, 0)

            # Measure properties of the regions
            IJ.run("Set Measurements...", "centroid area bounding area_fraction")
            IJ.run("Analyze Particles...", "display clear include summarize")

            # Get the results and add them to the ResultsTable
            current_results = ResultsTable.getResultsTable()
            results_table.incrementCounter()
            results_table.addValue("Image", file_name)
            results_table.addValue("Centroid_X", current_results.getValue("XM"))
            results_table.addValue("Centroid_Y", current_results.getValue("YM"))
            results_table.addValue("Centroid_Z", current_results.getValue("ZM"))
            results_table.addValue("Area", current_results.getValue("Area"))
            results_table.addValue("Bounding_Box", current_results.getValue("Bounding Area"))
            results_table.updateResults()

            # Add ROIs to the RoiManager
            roi_manager.addRoi(image.getRoi())

    # Save the ResultsTable and ROIs
    results_table.save(output_folder + "results.csv")
    roi_manager.runCommand("Save", output_folder + "rois.zip")

    # Close all images
    IJ.run("Close All")

    # Close the ResultsTable and RoiManager
    results_table.close()
    roi_manager.close()

