
from ij import IJ, ImagePlus
from ij.plugin import FFT

#Load the 3D PSF Image
image_path ="/home/shaswati/Documents/PSF/40x-1.4-banana/40X_PSF_OIL_OLYMPUS_05.vsi - C488.tif"
imp = IJ.openImage(image_path)

def analyze_psf(imp):
    # Load the 3D PSF image
    imp = IJ.openImage(image_path)
    fft = FFT()
    fft.run(imp)

    # The result of the FFT is stored in the ImagePlus object
    fft_result = imp.getProcessor()

    # Analyze the FFT result to determine the shape of the PSF
    # You can set custom thresholds based on your specific PSF images
    bent_threshold = 100
    elongated_threshold = 200

    max_magnitude = fft_result.getMax()

    if max_magnitude < bent_threshold:
        shape = "bent"
    elif max_magnitude < elongated_threshold:
        shape = "elongated"
    else:
        shape = "banana-shaped"

    return shape


# Analyze the PSF image
psf_shape = analyze_psf("/home/shaswati/Documents/PSF/40x-1.4-banana/40X_PSF_OIL_OLYMPUS_05.vsi - C488.tif")
print("The PSF shape is:", psf_shape)
