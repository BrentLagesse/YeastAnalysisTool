import numpy as np
from PIL import Image
import os
import skimage.exposure

'''Convert input images to RGB format in separate folders required by MRCNN

Arguments:
inputdirectory: Input directory containing images.
outputdirectory: Output directory to put new files in.
outputfile: Path to save comma-delimited file that will tell the neural network the image paths.'''
def preprocess_images(inputdirectory, mask_dir, outputdirectory, outputfile, verbose = False):
    if inputdirectory[-1] != "/":
        inputdirectory = inputdirectory + "/"
    if outputdirectory[-1] != "/":
        outputdirectory = outputdirectory + "/"

    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    output = open(outputfile, "w")
    output.write("ImageId, EncodedRLE" + "\n")
    output.close()

    for imagename in os.listdir(inputdirectory):
        if 'DIC' not in imagename:
            continue
        try:
            if verbose:
                print ("Preprocessing ", imagename)
            existing_files = os.listdir(mask_dir)
            if imagename in existing_files:   #skip this if we have a mask already
                continue
            image = np.array(Image.open(inputdirectory + imagename))
            if len(image.shape) > 2:
                image = image[:, :, 0]
            height = image.shape[0]
            width = image.shape[1]

            # Preprocessing operations
            image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)        #convert to 8 bit
            image = np.expand_dims(image, axis=-1)
            rgbimage = np.tile(image, 3)                          #convert to RGB

            imagename = imagename.split(".")[0]

            if not os.path.exists(outputdirectory + imagename):
                os.makedirs(outputdirectory + imagename)
                os.makedirs(outputdirectory + imagename + "/images/")
            rgbimage = Image.fromarray(rgbimage)
            rgbimage.save(outputdirectory + imagename + "/images/" + imagename + ".png")

            output = open(outputfile, "a")
            output.write(imagename + ", " + str(height) + " " + str(width) + "\n")
            output.close()
        except IOError:
            pass



