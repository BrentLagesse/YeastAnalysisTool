import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KERAS_BACKEND'] = 'tensorflow'

seed = 123
#from keras import backend as K

import numpy as np

np.random.seed(seed)


import random

random.seed(seed)

from PIL import Image
import skimage.transform
from skimage import img_as_ubyte
from cv2_rolling_ball import subtract_background_rolling_ball

import pandas as pd
import os

from . import my_functions as f

import time

#######################################################################################
## SET UP CONFIGURATION
from . import config

class BowlConfig(config.Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"

    IMAGE_RESIZE_MODE = "pad64"  ## tried to modfied but I am using other git clone
    ## No augmentation
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False  ## Not using this

    IMAGE_MIN_DIM = 512  # We scale small images up so that smallest side is 512
    IMAGE_MAX_DIM = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD = 0.2
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.001

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nuclei

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    USE_MINI_MASK = True

#######################################################################################

'''Run images through the pre-trained neural network.

Arguments:
test_path: Path where the images are stored (preprocess these using preprocess_images.py)
inputfile: Path of the comma-delimited file of images names.
outputfile: Path to write the comma-delimited run-length file to.
rescale: Set to True if rescale images before processing (saves time)
scale_factor: Multiplier to downsample images by
verbose: Verbose or not (true/false)'''
def predict_images(test_path, sample_submission, outputfilename, rescale = False, scale_factor = 2, verbose = True):
    inference_config = BowlConfig()
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    output = open(outputfilename, "w")
    output.truncate()
    output.write("ImageId, EncodedPixels\n")
    output.close()

    sample_submission = pd.read_csv(sample_submission)
    n_images = len(sample_submission.ImageId)
    if n_images == 0:   # loading tensorflow takes a long time.  Don't do it if we don't use it.
        return
    import tensorflow as tf
    from . import model as modellib

    tf.random.set_seed(seed)

    if test_path[-1] != "/":
        test_path = test_path + "/"
    
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../weights/deepretina_final.h5')

    if verbose:
        print("Loading weights from ", model_path)

    start_time = time.time()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model.load_weights(model_path, by_name=True)


    for i in np.arange(n_images):
        start_time = time.time()
        image_id = sample_submission.ImageId[i]
        if verbose:
            print('Start detect', i, '  ', image_id)
        ##Set seeds for each image, just in case..
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        ## Load the image
        image_path = os.path.join(test_path, image_id, 'images', image_id + '.tif')
        original_image = np.array(Image.open(image_path))



        if rescale:
            height = original_image.shape[0]
            width = original_image.shape[1]
            original_image = skimage.transform.resize(original_image, output_shape=(height // scale_factor,
                                                                                    width // scale_factor),
                                                                                    preserve_range=True)

        ####################################################################
        ## This is needed for the stage 2 image that has only one channel
        if len(original_image.shape) < 3:
            original_image = img_as_ubyte(original_image)
            original_image = np.expand_dims(original_image, 2)
            original_image = original_image[:, :, [0, 0, 0]]  # flip r and b
        ####################################################################
        original_image = original_image[:, :, :3]
        #original_image, background = subtract_background_rolling_ball(original_image, 50, light_background=False,
        #                                                                   use_paraboloid=False, do_presmooth=True)
        ## Make prediction for that image
        results = model.detect([original_image], verbose=0)

        ## Proccess prediction into rle
        pred_masks = results[0]['masks']
        scores_masks = results[0]['scores']
        class_ids = results[0]['class_ids']

        if len(class_ids):  ## Some objects are detected
            ImageId_batch, EncodedPixels_batch, _ = f.numpy2encoding(pred_masks, image_id, scores=scores_masks,
                                                                     dilation=True)
            f.write2csv(outputfilename, ImageId_batch, EncodedPixels_batch)
        else:
            pass

        if verbose:
            print("Completed in", time.time() - start_time)

