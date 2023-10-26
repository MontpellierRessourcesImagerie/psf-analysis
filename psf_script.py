

def part4():
    # Import necessary Fiji classes
    from ij import IJ
    from ij import ImagePlus
    from ij.measure import ResultsTable
    from ij.plugin.frame import RoiManager
    from inra.ijpb.label import LabelImages
    from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling3D
    import os
    from ij.plugin import Thresholder
    from ij import ImageStack
    from ij.process import ImageConverter

    # Set the path to your 3D TIFF images
    image_folder  = "/media/benedetti/7A1CDD9C53CA4C38/PSF/40x-1.4-banana"
    output_folder = "/media/benedetti/7A1CDD9C53CA4C38/PSF/40x-1.4-banana/Output_PSF.csv"

    # Initialize a ResultsTable to store measurement results
    results_table = ResultsTable()

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, file_name)
        if file_name.lower().endswith(".tif"):
            # Open the 3D image
            image = IJ.openImage(full_path)

            IJ.run(image, "Subtract Background...", "rolling=50 stack")

            image.show()

            IJ.run("FeatureJ Laplacian", "compute smoothing=0.25 stack")
            lapresult = IJ.getImage()

            mask = lapresult.duplicate()
            lapresult.close()

            # ImageConverter.setDoScaling(True)
            IJ.setAutoThreshold(mask, "Default light stack")
            IJ.run(mask, "Convert to Mask", "Background=light")
            mask.getProcessor().invert()

            # Label the connected components using FloodFillRegionComponentsLabeling
            ffrcl = FloodFillRegionComponentsLabeling3D(26, 8)
            # labeled_image = FloodFillRegionComponentsLabeling.labelComponents(image, 8, 8, 0)
            newstack = ffrcl.computeLabels(mask.getStack(), 255)

            labeled_image = ImagePlus("labeled-image", newstack)
            labeled_image.show()
            return

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
    from ij import ImagePlus
    from ij.plugin.frame import RoiManager
    from inra.ijpb.label.conncomp import FloodFillRegionComponentsLabeling3D
    from inra.ijpb.label import LabelImages
    from inra.ijpb.plugins import AnalyzeRegions3D
    import os

    # Set the path to your 3D TIFF images
    image_folder = "/home/shaswati/Documents/PSF/40x-1.4-banana"
   

    # Initialize a RoiManager to store ROIs
    roi_manager = RoiManager()

    # Process each 3D TIFF image in the folder
    for file_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, file_name)
        if file_name.lower().endswith(".tif"):
                # Open the 3D image
                image = IJ.openImage(full_path)

                IJ.run(image, "Subtract Background...", "rolling=50 stack")
                image.show()
                rsl_good = image.getTitle()

                lapresult = IJ.getImage()

                mask = lapresult.duplicate()
                lapresult.close()

                # ImageConverter.setDoScaling(True)
                IJ.setAutoThreshold(mask, "Yen light stack")
                IJ.run(mask, "Convert to Mask", "Background=light stack")
                #mask.getProcessor().invert()
                mask.show()

                # Use FloodFillRegionComponentsLabeling3D (FFRC) to label the connected components
                ffrcl = FloodFillRegionComponentsLabeling3D()
                labeled_image = ffrcl.computeLabels(mask.getStack(), 26)  # 26-connectivity and 16-bit image

                # Use LabelImages to remove border regions
                LabelImages.removeBorderLabels(labeled_image)

                # Analyze regions using AnalyzeRegions3D
                clean_image = ImagePlus("PSF_rsl", labeled_image)
                IJ.saveAs(clean_image,"Tiff",os.path.join(image_folder,rsl_good +"_mask.tif"))
                analyze_regions = AnalyzeRegions3D()
                rsl= analyze_regions.process(clean_image)
                rsl.saveAs(os.path.join(image_folder,rsl_good +".csv"))
                
                
   

                # Close all images
                IJ.run("Close All")

                # Close the RoiManager
                roi_manager.close()

def main():
    part5()

main()

