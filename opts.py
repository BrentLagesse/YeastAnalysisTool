############# INPUTS AND OUTPUTS #############
# Input directory of images to be segmented
input_directory = "./inputs/"

# Output directory to save masks to
output_directory = "./outputs/"
##############################################

################## OPTIONS ###################
# Set to true to rescale the input images to reduce segmentation time
rescale = False
scale_factor = 2          # Factor to downsize images by if rescale is True

# Set to true to save preprocessed images as input to neural network (useful for debugging)
save_preprocessed = False

# Set to true to save a compressed RLE version of the masks for sharing
save_compressed = False

# Set to true to save the full masks
save_masks = True

# Set to true to have the neural network print out its segmentation progress as it proceeds
verbose = True

# Set to true to output ImageJ-compatible masks
output_imagej = False
##############################################
