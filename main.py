# from curses import window
import pytz
from tkinter import *
import customtkinter
from tkinter import filedialog
from tkinter.ttk import *
from functools import partial
from scipy.spatial import distance as dist
import math

import opts as opt
import os
import shutil

import csv
import cv2
import numpy as np
from PIL import ImageTk,Image
import PIL
import skimage.morphology
import skimage.exposure
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from mrc import DVFile
from mrcnn.my_inference import predict_images
from mrcnn.preprocess_images import preprocess_images
from mrcnn.convert_to_image import convert_to_image, convert_to_imagej
from enum import Enum
from cv2_rolling_ball import subtract_background_rolling_ball
import stats
from export import export_to_csv_file

input_dir = opt.input_directory
output_dir = opt.output_directory
ignore_btn = None

current_image = None
current_cell = None

outline_dict = dict()  # store the outlines so I don't have to reprocess them

image_dict = dict()    # image -> list of cell ids
cp_dict = dict()        # (image, id) -> cp

n = 0

class Contour(Enum):
    CONTOUR = 0
    CONVEX = 1
    CIRCLE = 2


class CellPair:
    def __init__(self, image_name, id):
        self.is_correct = True
        self.image_name = image_name
        print("Image name", image_name)
        self.id = id
        self.nuclei_count = 1
        self.red_dot_count = 1
        self.red_dot_distance = 0
        self.cyan_dot_count = 1
        self.green_dot_count = 1
        self.ground_truth = False
        self.nucleus_intensity = dict()
        self.nucleus_total_points = 0
        self.cell_intensity = dict()
        self.cell_total_points = 0
        self.ignored = False
        self.mcherry_line_gfp_intensity = 0

    def set_red_dot_distance(self, d):
        self.red_dot_distance = d

    def get_base_name(self):
        print("imagetest:",self.image_name)
        return self.image_name.split('_PRJ')[0]

    def get_DIC(self, use_id=False, outline=True, segmented=False, main_img=False):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if main_img:
            return Image.open(output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '.tif')
        if segmented:
            return Image.open(output_dir + 'segmented/'+ self.get_base_name() + '_PRJ' + '-' + str(self.id) + outlinestr + '.tif')
        if use_id:
            image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '-' + str(self.id) + outlinestr + '.tif'
            return Image.open(image_loc)
        else:
            # look for dv file,
            # open dv file if exists,
            # return the appropriate image from the stack (actual image)
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                img = Image.fromarray(image[0])
                return img
            else:
                image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + outlinestr + '.tif'
                return Image.open(image_loc)



    def get_DAPI(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        # check if there are .dv files and use them first
        if not use_id:
            image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '-' + str(self.id) + outlinestr + '.tif'
            return Image.open(image_loc)
        else:
            #return self.get_base_name() + '_PRJ' + '_w435' + outlinestr + '.tif'
            # look for dv file,
            # open dv file if exists,
            # return the appropriate image from the stack (actual image)
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                img = Image.fromarray(image[1])
                return img
            else:
                image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + outlinestr + '.tif'
                return Image.open(image_loc)

    def get_GFP(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        # check if there are .dv files and use them first
        if use_id:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w625' + '-' + str(self.id)  + outlinestr + '.tif'
            #image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '_w525' + '-' + str(self.id) + outlinestr + '.tif'
            #return Image.open(image_loc)
            return self.get_base_name() + '_PRJ' + '-' + str(self.id) + outlinestr + '.tif'
        else:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w625' + outlinestr + '.tif'
            # look for dv file,
            # open dv file if exists,
            # return the appropriate image from the stack (actual image)
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                img = Image.fromarray(image[2])
                return img
            else:
                #image_loc = output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '_w525' + outlinestr + '.tif'
                #return Image.open(image_loc)
                return self.get_base_name() + '_PRJ' + outlinestr + '.tif'

    def set_GFP_Nucleus_Intensity(self, contour_type, val, total_points):
        self.nucleus_intensity[contour_type] = val
        self.nucleus_total_points = total_points

    def set_GFP_Cell_Intensity(self, val, total_points):
        self.cell_intensity  = val
        self.cell_total_points = total_points

    def get_GFP_Nucleus_Intensity(self, contour_type):
        if self.nucleus_intensity.get(contour_type) == None:
            return (0, 0)
        if self.nucleus_intensity[contour_type]  == 0:   # this causes an error if nothing has been set.  This is expected
            print ("Intensity is 0, this is unlikely")
        return self.nucleus_intensity[contour_type] , self.nucleus_total_points

    def get_GFP_Cell_Intensity(self):
        if self.cell_intensity  == 0:
            print ("Intensity is 0, this is unlikely")
        return self.cell_intensity, self.cell_total_points

    def set_mcherry_line_GFP_intensity(self, intensity):
        self.mcherry_line_gfp_intensity = intensity

    def get_mcherry_line_GFP_intensity(self):
        return self.mcherry_line_gfp_intensity

    def get_mCherry(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            #return output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '_w625' + '-' + str(self.id) + outlinestr + '.tif'
            #return Image.open(image_loc)
            return self.get_base_name() + '_PRJ' +  '-' + str(self.id) + outlinestr + '.tif'
        else:
            # look for dv file,
            # open dv file if exists,
            # return the appropriate image from the stack (actual image)
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                img = Image.fromarray(image[3])
                return img
            else:
                #return output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '_w625' + outlinestr + '.tif'
                #return Image.open(image_loc)
                return self.get_base_name() + '_PRJ' + outlinestr + '.tif'

    #TODO: Remove is Matt says we will never get this one
    def get_CFP(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '-' + str(self.id)  + outlinestr + '.tif'
        else:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w435' + outlinestr + '.tif'
            # look for dv file,
            # open dv file if exists,
            # return the appropriate image from the stack (actual image)
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                img = Image.fromarray(image[1])
                return img
            else:
                return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + outlinestr + '.tif'


    def get_member_variables(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def get_values_as_csv(self):
        #get all the member variables
        members = self.get_member_variables()
        # get my variables
        values = [getattr(self, member) for member in members]  # get my values
        return ','.join(values)

    def classify(self, id=False):
        pass

    def update(self, is_correct, nuclei_count, red_dot_count, red_dot_distance):
        self.is_correct = is_correct
        self.nuclei_count = nuclei_count
        self.red_dot_count = red_dot_count
        self.red_dot_distance = red_dot_distance

    def set_ignored(self, enabled):
        self.ignored = enabled

    def get_ignored(self):
        return self.ignored

def load_result(cell_id=0):
    global image_dict
    #TODO:  if cell_id is negative, loop back around
    pass
    #

def export_to_csv():
    global image_dict
    global cp_dict
    global drop_ignored
    export_to_csv_file(data,window, image_dict, cp_dict, drop_ignored)



def segment_images():
    global image_dict
    #TODO:  ask user if they want to refresh segmentation
    global window
    global output_dir 
    global input_dir 
    global img_label
    global DIC_label
    global DAPI_label
    global mCherry_label
    global GFP_label
#    global CFP_label
    global ID_label
    global outline_dict
    global cp_dict

    cp_dict.clear()
    image_dict.clear()
    print("cp_dict", cp_dict)

    print("kernel size", int(kernel_size_input))
    print("kernel diviation",int(kernel_deviation_input))
    print("mcher",int(mcherry_line_width_input))
    print('usechace',use_cache.get())
    print('use_spc110', use_spc110.get())
    print('arrested', choice_var)


    if input_dir[-1] != "/":
        input_dir = input_dir + "/"
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"

    if output_dir != '' and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.isdir(output_dir):

        preprocessed_image_directory = output_dir + "preprocessed_images/"
        preprocessed_image_list = output_dir + "preprocessed_images_list.csv"
        rle_file = output_dir + "compressed_masks.csv"
        output_mask_directory = output_dir + "masks/"
        output_imagej_directory = output_dir + "imagej/"

        # Preprocess the images
        if opt.verbose:
            print("\nPreprocessing your images...")
        #TODO:  put everything in separate directories -- list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
        preprocess_images(input_dir,
                          output_mask_directory,
                          preprocessed_image_directory,
                          preprocessed_image_list,
                          verbose=opt.verbose,
                          use_cache = use_cache.get())

        if opt.verbose:
            print("\nRunning your images through the neural network...")
        predict_images(preprocessed_image_directory,
                       preprocessed_image_list,
                       rle_file,
                       rescale=opt.rescale,
                       scale_factor=opt.scale_factor,
                       verbose=opt.verbose)

        if opt.save_masks == True:
            if opt.verbose:
                print("\nSaving the masks...")

            if opt.output_imagej == True:
                convert_to_image(rle_file,
                                 output_mask_directory,
                                 preprocessed_image_list,
                                 rescale=opt.rescale,
                                 scale_factor=opt.scale_factor,
                                 verbose=opt.verbose)

                convert_to_imagej(output_mask_directory,
                                  output_imagej_directory)
            else:
                convert_to_image(rle_file,
                                 output_mask_directory,
                                 preprocessed_image_list,
                                 rescale=opt.rescale,
                                 scale_factor=opt.scale_factor,
                                 verbose=opt.verbose)

        os.remove(preprocessed_image_list)

        if not opt.save_preprocessed:
            shutil.rmtree(preprocessed_image_directory)

        if not opt.save_compressed:
            os.remove(rle_file)

        if not opt.save_masks:
            shutil.rmtree(output_mask_directory)

    def get_neighbor_count(seg_image, center, radius=1, loss=0):
        #TODO:  account for loss as distance gets larger
        neighbor_list = list()
        neighbors = seg_image[center[0] - radius:center[0] + radius + 1, center[1] - radius:center[1] + radius + 1]
        for x, row in enumerate(neighbors):
            for y, val in enumerate(row):
                if (x, y) != (radius, radius) and int(val) != 0 and int(val) != int(seg_image[center[0], center[1]]):
                    neighbor_list.append(val)
        return neighbor_list



    for image_name in os.listdir(input_dir):
        if '_PRJ' not in image_name:
            continue
        extspl = os.path.splitext(image_name)
        if len(extspl) != 2 or extspl[1] != '.dv':  # ignore files that aren't dv
            continue
        seg = None
        image_dict[image_name] = list()
        #cp = CellPair(output_dir + 'segmented/' + image_name.split('.')[0] + '.tif', output_dir + 'masks/' + image_name.split('.')[0] + '.tif')
        exist_check = os.path.exists(output_dir + 'masks/' + image_name.split('.')[0]+ '-cellpairs' + '.tif')
        if exist_check and use_cache.get():
            seg = np.array(Image.open(output_dir + 'masks/' + image_name.split('.')[0]+ '-cellpairs' + '.tif'))
            outlines = np.zeros(seg.shape)
        else:


            segmentation_name = output_dir + 'masks/' + image_name
            seg_ext = os.path.splitext(segmentation_name)
            segmentation_name = seg_ext[0] + '.tif'
            #image_dict[image_name] = segmentation_name
            #Load the original raw image and rescale its intensity values
            #image = np.array(Image.open(input_dir + image_name))
            f = DVFile(input_dir + image_name)
            im = f.asarray()
            image = Image.fromarray(im[0])
            #image = np.float32(image)
            image = skimage.exposure.rescale_intensity(np.float32(image), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            debug_image = image

            # Convert the image to an RGB image, if necessary
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                image = np.expand_dims(image, axis=-1)
                image = np.tile(image, 3)

            # Open the segmentation file    # TODO -- make it show it is choosing the correct segmented
            seg = np.array(Image.open(segmentation_name))   #TODO:  on first run, this can't find outputs/masks/M***.tif'

            #TODO:   If G1 Arrested, we don't want to merge neighbors and ignore non-budding cells
            #choices = ['Metaphase Arrested', 'G1 Arrested']
            outlines = np.zeros(seg.shape)
            if choice_var == 'Metaphase Arrested':
                # Create a raw file to store the outlines

                ignore_list = list()
                single_cell_list = list()
                # merge cell pairs
                neighbor_count = dict()
                closest_neighbors = dict()
                for i in range(1, int(np.max(seg) + 1)):
                    cells = np.where(seg == i)
                    #examine neighbors
                    neighbor_list = list()
                    for cell in zip(cells[0], cells[1]):
                        #TODO:  account for going over the edge without throwing out the data

                        try:
                            neighbor_list = get_neighbor_count(seg, cell, 3)
                            #neighbor_list.append(seg[cell[0]-1][cell[1]-1])  #top left
                            #neighbor_list.append(seg[cell[0]][cell[1]-1])  #top
                            #neighbor_list.append(seg[cell[0]+1][cell[1]-1])  #top right
                            #neighbor_list.append(seg[cell[0]-1][cell[1]])  #left
                            #neighbor_list.append(seg[cell[0]+1][cell[1]])  #right
                            #neighbor_list.append(seg[cell[0]-1][cell[1]+1])  #bottom left
                            #neighbor_list.append(seg[cell[0]][cell[1]+1])  #bottom
                            #neighbor_list.append(seg[cell[0]+1][cell[1]+1])  #bottom right
                        except:
                            continue

                        for neighbor in neighbor_list:
                            if int(neighbor) == i or int(neighbor) == 0:
                                continue
                            if neighbor in neighbor_count:
                                neighbor_count[neighbor] += 1
                            else:
                                neighbor_count[neighbor] = 1

                    sorted_dict = {k: v for k, v in sorted(neighbor_count.items(), key=lambda item: item[1])}
                    #v = list(neighbor_count.values())
                    #k = list(neighbor_count.keys())
                    if len(sorted_dict) == 0:
                        print('found single cell at: ' + str(cell))
                        single_cell_list.append(int(i))
                    else:
                            #closest_neighbor = k[v.index(max(v))]
                        if len(sorted_dict) == 1:
                            closest_neighbors[i] = list(sorted_dict.items())[0][0]
                        else:
                            top_val = list(sorted_dict.items())[0][1]
                            second_val = list(sorted_dict.items())[1][1]
                            if second_val > 0.5 * top_val:    # things got confusing, so we throw it and its neighbor out
                                single_cell_list.append(int(i))
                                for cluster_cell in neighbor_count:
                                    single_cell_list.append(int(cluster_cell))
                            else:
                                closest_neighbors[i] = list(sorted_dict.items())[0][0]

                    #reset for the next cell
                    neighbor_count = dict()
                #TODO:  Examine the spc110 dots and make closest dots neighbors
                resolve_cells_using_spc110 = use_spc110.get()
                lines_to_draw = dict()
                if resolve_cells_using_spc110:

                    # open the mcherry
                    #TODO: open mcherry from dv stack

                    # basename = image_name.split('_R3D_REF')[0]
                    # mcherry_dir = input_dir + basename + '_PRJ_TIFFS/'
                    # mcherry_image_name = basename + '_PRJ' + '_w625' + '.tif'
                    # mcherry_image = np.array(Image.open(mcherry_dir + mcherry_image_name))
                    f = DVFile(input_dir + image_name)
                    mcherry_image = f.asarray()[3]

                    mcherry_image = skimage.exposure.rescale_intensity(mcherry_image.astype(np.float32), out_range=(0, 1))
                    mcherry_image = np.round(mcherry_image * 255).astype(np.uint8)

                    # Convert the image to an RGB image, if necessary
                    if len(mcherry_image.shape) == 3 and mcherry_image.shape[2] == 3:
                        pass
                    else:
                        mcherry_image = np.expand_dims(mcherry_image, axis=-1)
                        mcherry_image = np.tile(mcherry_image, 3)
                    # find contours
                    mcherry_image_gray = cv2.cvtColor(mcherry_image, cv2.COLOR_RGB2GRAY)
                    mcherry_image_gray, background = subtract_background_rolling_ball(mcherry_image_gray, 50,
                                                                                       light_background=False,
                                                                                       use_paraboloid=False,
                                                                                       do_presmooth=True)

                    debug = False
                    if debug:
                        plt.figure(dpi=600)
                        plt.title("mcherry")
                        plt.imshow(mcherry_image_gray, cmap='gray')
                        plt.show()

                    #mcherry_image_gray = cv2.GaussianBlur(mcherry_image_gray, (1, 1), 0)
                    mcherry_image_ret, mcherry_image_thresh = cv2.threshold(mcherry_image_gray, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
                    mcherry_image_cont, mcherry_image_h = cv2.findContours(mcherry_image_thresh, 1, 2)

                    if debug:
                        cv2.drawContours(image, mcherry_image_cont, -1, 255, 1)
                        plt.figure(dpi=600)
                        plt.title("ref image with contours")
                        plt.imshow(image, cmap='gray')
                        plt.show()


                    #921,800

                    min_mcherry_distance = dict()
                    min_mcherry_loc = dict()   # maps an mcherry dot to its closest mcherry dot in terms of cell id
                    for cnt1 in mcherry_image_cont:
                        try:
                            contourArea = cv2.contourArea(cnt1)
                            if contourArea > 100000:   #test for the big box, TODO: fix this to be adaptive
                                print('threw out the bounding box for the entire image')
                                continue
                            M1 = cv2.moments(cnt1)
                            # These are opposite of what we would expect
                            c1y = int(M1['m10'] / M1['m00'])
                            c1x = int(M1['m01'] / M1['m00'])


                        except:  #no moment found
                            continue
                        c_id = int(seg[c1x][c1y])
                        if c_id == 0:
                            continue
                        for cnt2 in mcherry_image_cont:
                            try:
                                M2 = cv2.moments(cnt2)
                                # find center of each contour
                                c2y = int(M2['m10'] / M2['m00'])
                                c2x = int(M2['m01'] / M2['m00'])

                                

                            except:
                                continue #no moment found
                            if int(seg[c2x][c2y]) == 0:
                                continue
                            if seg[c1x][c1y] == seg[c2x][c2y]:   #these are ihe same cell already -- Maybe this is ok?  TODO:  Figure out hwo to handle this because some of the mcherry signals are in the same cell
                                continue
                            # find the closest point to each center
                            d = math.sqrt(pow(c1x - c2x, 2) + pow(c1y - c2y, 2))
                            if min_mcherry_distance.get(c_id) == None:
                                min_mcherry_distance[c_id] = d
                                min_mcherry_loc[c_id] = int(seg[c2x][c2y])
                                lines_to_draw[c_id] = ((c1y,c1x), (c2y, c2x))
                            else:
                                if d < min_mcherry_distance[c_id]:
                                    min_mcherry_distance[c_id] = d
                                    min_mcherry_loc[c_id] = int(seg[c2x][c2y])
                                    lines_to_draw[c_id] = ((c1y, c1x), (c2y, c2x))  #flip it back here
                                elif d == min_mcherry_distance[c_id]:
                                    print('This is unexpected, we had two mcherry red dots in cells {} and {} at the same distance from ('.format(seg[c1x][c1y], seg[c2x][c2y]) + str(min_mcherry_loc[c_id]) + ', ' + str((c2x, c2y)) + ') to ' + str((c1x, c1y)) + ' at a distance of ' + str(d))







                for k, v in closest_neighbors.items():
                    if v in closest_neighbors:      # check to see if v could be a mutual pair
                        if int(v) in ignore_list:    # if we have already paired this one, throw it out
                            single_cell_list.append(int(k))
                            continue

                        if closest_neighbors[int(v)] == int(k) and int(k) not in ignore_list:  # closest neighbors are reciprocal
                            #TODO:  set them to all be the same cell
                            to_update = np.where(seg == v)
                            ignore_list.append(int(v))
                            if resolve_cells_using_spc110:
                                if int(v) in min_mcherry_loc:    #if we merge them here, we don't need to do it with mcherry
                                    del min_mcherry_loc[int(v)]
                                if int(k) in min_mcherry_loc:
                                    del min_mcherry_loc[int(k)]
                            for update in zip(to_update[0], to_update[1]):
                                seg[update[0]][update[1]] = k

                        elif int(k) not in ignore_list and not resolve_cells_using_spc110:
                            single_cell_list.append(int(k))


                    elif int(k) not in ignore_list and not resolve_cells_using_spc110:
                        single_cell_list.append(int(k))

                if resolve_cells_using_spc110:
                    for c_id, nearest_cid in min_mcherry_loc.items():
                        if int(c_id) in ignore_list:    # if we have already paired this one, ignore it
                            continue
                        if int(nearest_cid) in min_mcherry_loc:  #make sure teh reciprocal exists
                            if min_mcherry_loc[int(nearest_cid)] == int(c_id) and int(c_id) not in ignore_list:   # if it is mutual
                                print('added a cell pair in image {} using the mcherry technique {} and {}'.format(image_name, int(nearest_cid),
                                                                                                       int(c_id)))
                                if int(c_id) in single_cell_list:
                                    single_cell_list.remove(int(c_id))
                                if int(nearest_cid) in single_cell_list:
                                    single_cell_list.remove(int(nearest_cid))
                                to_update = np.where(seg == nearest_cid)
                                closest_neighbors[int(c_id)] = int(nearest_cid)
                                ignore_list.append(int(nearest_cid))
                                for update in zip(to_update[0], to_update[1]):
                                    seg[update[0]][update[1]] = c_id
                            elif int(c_id) not in ignore_list:
                                print('could not add cell pair because cell {} and cell {} were not mutually closest'.format(nearest_cid, int(v)))
                                single_cell_list.append(int(k))

                # remove single cells or confusing cells
                for cell in single_cell_list:
                    seg[np.where(seg == cell)] = 0.0


                # only merge if two cells are both each others closest neighbors
                    # otherwise zero them out?
                # rebase segment count
                to_rebase = list()
                for k, v in closest_neighbors.items():
                    if k in ignore_list or k in single_cell_list:
                        continue
                    else:
                        to_rebase.append(int(k))
                to_rebase.sort()

                for i, x in enumerate(to_rebase):
                    seg[np.where(seg == x)] = i + 1

                # now seg has the updated masks, so lets save them so we don't have to do this every time
                seg_image = Image.fromarray(seg)
                seg_image.save(output_dir + 'masks/' + image_name.split('.')[0]+ '-cellpairs' + '.tif')
            else:   #g1 arrested
                pass

        for i in range(1, int(np.max(seg)) + 1):
            image_dict[image_name].append(i)

        base_image_name = image_name.split('_PRJ')[0]
        for images in os.listdir(input_dir):
            # don't overlay if it isn't the right base image
            if base_image_name not in images:
                continue
            extspl = os.path.splitext(images)
            if len(extspl) != 2 or extspl[1] != '.dv':  # ignore files that aren't dv
                continue
            if_g1 = ''
            #if choice_var.get() == 'G1 Arrested':   #if it is a g1 cell, do we really need a separate type of file?
            #    if_g1 = '-g1'
            tif_image = images.split('.')[0] + if_g1 + '.tif'
            if os.path.exists(output_dir + 'segmented/' + tif_image) and use_cache.get():
                continue
            to_open = input_dir + images
            if os.path.isdir(to_open):
                continue
            #image = np.array(Image.open(to_open))
            f = DVFile(to_open)
            im = f.asarray()
            image = Image.fromarray(im[0])
            image = skimage.exposure.rescale_intensity(np.float32(image), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            # Convert the image to an RGB image, if necessary
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                image = np.expand_dims(image, axis=-1)
                image = np.tile(image, 3)

            # Iterate over each integer in the segmentation and save the outline of each cell onto the outline file
            for i in range(1, int(np.max(seg) + 1)):
                tmp = np.zeros(seg.shape)
                tmp[np.where(seg == i)] = 1
                tmp = tmp - skimage.morphology.binary_erosion(tmp)
                outlines += tmp




            # Overlay the outlines on the original image in green
            image_outlined = image.copy()
            image_outlined[outlines > 0] = (0, 255, 0)

            # Display the outline file
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image_outlined)

            # debugging to see where the mcherry signals connect
            for k, v in lines_to_draw.items():
                start, stop = v
                cv2.line(image_outlined, start, stop, (255,0,0), 1)
                #txt = ax.text(start[0], start[1], str(start), size=12)
                #txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                #txt = ax.text(stop[0], stop[1], str(stop), size=12)
                #txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])


            # iterate over each cell pair and add an ID to the image
            for i in range(1, int(np.max(seg) + 1)):
                loc = np.where(seg == i)
                if len(loc[0]) > 0:
                    txt = ax.text(loc[1][0], loc[0][0], str(i), size=12)
                    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                else:
                    print('could not find cell id ' + str(i))

            fig.savefig(output_dir + 'segmented/' + tif_image, dpi=600, bbox_inches='tight', pad_inches=0)

            #plt.show()

        #TODO:  Combine the two iterations over the input directory images

        # This is where we overlay what we learned in the DIC onto the other images

        filter_dir = input_dir  + base_image_name + '_PRJ_TIFFS/'
        f = DVFile(input_dir + image_name)
        for image_num in range(1,3):
            # images = os.path.split(full_path)[1]  # we start in separate directories, but need to end up in the same one
            # # don't overlay if it isn't the right base image
            # if base_image_name not in images:
            #     continue
            # extspl = os.path.splitext(images)
            # if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't dv
            #     continue
            # #tif_image = images.split('.')[0] + '.tif'
            #
            # if os.path.isdir(full_path):
            #     continue
            image = np.array(f.asarray()[image_num])
            image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            # Convert the image to an RGB image, if necessary
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                image = np.expand_dims(image, axis=-1)
                image = np.tile(image, 3)

            outlines = np.zeros(seg.shape)
            # Iterate over each integer in the segmentation and save the outline of each cell onto the outline file
            for i in range(1, int(np.max(seg) + 1)):
                tmp = np.zeros(seg.shape)
                tmp[np.where(seg == i)] = 1
                tmp = tmp - skimage.morphology.binary_erosion(tmp)
                outlines += tmp
            
            # Overlay the outlines on the original image in green
            image_outlined = image.copy()
            image_outlined[outlines > 0] = (0, 255, 0)

            # Iterate over each integer in the segmentation and save the outline of each cell onto the outline file
            for i in range(1, int(np.max(seg) + 1)):
                cell_tif_image = tif_image.split('.')[0] + '-' + str(i) + '.tif'
                no_outline_image = tif_image.split('.')[0] + '-' + str(i) + '-no_outline.tif'
                # cell_tif_image = images.split('.')[0] + '-' + str(i) + '.tif'
                # no_outline_image = images.split('.')[0] + '-' + str(i) + '-no_outline.tif'






                a = np.where(seg == i)   # somethin bad is happening when i = 4 on my tests
                min_x = max(np.min(a[0]) - 1, 0)
                max_x = min(np.max(a[0]) + 1, seg.shape[0])
                min_y = max(np.min(a[1]) - 1, 0)
                max_y = min(np.max(a[1]) + 1, seg.shape[1])

                # a[0] contains the x coords and a[1] contains the y coords
                # save this to use later when I want to calculate cellular intensity

                #convert from absolute location to relative location for later use


                if not os.path.exists(output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline')  or not use_cache.get():
                    with open(output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline', 'w') as csvfile:
                        csvwriter = csv.writer(csvfile, lineterminator='\n')
                        csvwriter.writerows(zip(a[0] - min_x, a[1] - min_y))






                cellpair_image = image_outlined[min_x: max_x, min_y:max_y]
                not_outlined_image = image[min_x: max_x, min_y:max_y]
                if not os.path.exists(output_dir + 'segmented/' + cell_tif_image) or not use_cache.get():  # don't redo things we already have
                    plt.imsave(output_dir + 'segmented/' + cell_tif_image, cellpair_image, dpi=600, format='TIFF')
                    plt.clf()
                if not os.path.exists(output_dir + 'segmented/' + no_outline_image) or not use_cache.get():  # don't redo things we already have
                    plt.imsave(output_dir + 'segmented/' + no_outline_image, not_outlined_image, dpi=600, format='TIFF')
                    plt.clf()

    # if the image_dict is empty, then we didn't get anything interesting from the directory
    print("image_dict123", image_dict)
    if len(image_dict) > 0:
        k, v = list(image_dict.items())[0]
        print("displaycell",k,v[0])
        display_cell(k, v[0])
    #else: show error message

def ignore(image, id):
    global ignore_btn
    global cp_dict
    cp_dict[(image, id)].set_ignored(not cp_dict[(image, id)].get_ignored())
    if cp_dict[(image, id)].get_ignored():
        ignore_btn.configure(text = 'ENABLE')
    else:
        ignore_btn.configure(text = 'IGNORE')

    # attempt to get distance
    #testimg = cv2.imread(image_loc, cv2.IMREAD_UNCHANGED)



#TODO:  Deal with resizing
def on_resize(event):
    try:
        w = width
        h = height
    except:
        return


def display_cell(image, id):
    global ignore_btn
    global current_image
    global current_cell
    global export_btn
    global drop_ignored_checkbox
    global window
    global data
    export_btn['state'] = NORMAL
    drop_ignored_checkbox['state'] = NORMAL

    current_image = image
    current_cell = id

    win_width = window.winfo_width()
    win_height = window.winfo_height()
    max_id = len(image_dict[image])
    if max_id == 0:
        print('No cells found in this image')
        return
    if id < 1:
        id = max_id
    if id > max_id:
        id = 1
    ID_label.configure(text='Cell ID:  ' + str(id))
    img_title_label.configure(text=image)
    print("displayImagename", image, cp_dict)
    cp = cp_dict.get((image, id))
    print("cp123", cp)
    if cp == None:
        cp = CellPair(image, id)
        cp_dict[(image, id)] = cp
    main_size_x = int(0.5 * win_width)
    main_size_y = int(0.5 * win_height)
    cell_size_x = int(0.23*main_size_x)
    cell_size_y = int(0.23*main_size_x)

    im_cherry, im_gfp = stats.get_stats(cp,data)
    #image_loc = output_dir + 'segmented/' + cp.get_DIC()  #TODO:  This messes up with the _w50 naming
    im = cp.get_DIC(main_img=True)
    width, height = im.size
    if height > width:
        scale = float(width)/float(height)
    else:
        scale = float(height) / float(width)
    im = im.resize((int(scale * main_size_x), main_size_y),  resample=Image.NEAREST)
    img = ImageTk.PhotoImage(im)
    img_label.configure(image=img)
    img_label.image = img


    #TODO: Do not show DIC and DAPI if file is video/live cell images
    #if os.path.getsize(input_dir + image) < 9000000:                #if file size is less than 9MB show DIC and DAPI
    im = cp.get_DIC(segmented=True)
    width, height = im.size
    if height > width:
        scale = float(width) / float(height)
        x_scaled = (int(scale * cell_size_x))
        y_scaled = cell_size_y
    else:
        scale = float(height) / float(width)
        x_scaled = cell_size_x
        y_scaled = int(scale * cell_size_y)
    im = im.resize((x_scaled, y_scaled), resample=Image.NEAREST)
    img = ImageTk.PhotoImage(im)
    DIC_label.configure(image=img)
    DIC_label.image = img
    DIC_label_text.configure(text="DIC")

    im = cp.get_DAPI()
    width, height = im.size
    if height > width:
        scale = float(width) / float(height)
        x_scaled = (int(scale * cell_size_x))
        y_scaled = cell_size_y
    else:
        scale = float(height) / float(width)
        x_scaled = cell_size_x
        y_scaled = int(scale * cell_size_y)
    im = im.resize((x_scaled, y_scaled), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im)
    DAPI_label.configure(image=img)
    DAPI_label.image = img
    DAPI_label_text.configure(text="DAPI")

    #Live Cell Images only supports mcherry and GFP
    image_loc = output_dir + 'segmented/' + cp.get_mCherry(use_id=True)
    im = im_cherry
    width, height = im.size
    if height > width:
        scale = float(width) / float(height)
        x_scaled = (int(scale * cell_size_x))
        y_scaled = cell_size_y
    else:
        scale = float(height) / float(width)
        x_scaled = cell_size_x
        y_scaled = int(scale * cell_size_y)
    im = im.resize((x_scaled, y_scaled), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im)
    mCherry_label.configure(image=img)
    mCherry_label.image = img
    mCherry_label_text.configure(text="mCherry")




    image_loc = output_dir + 'segmented/' + cp.get_mCherry(use_id=True, outline=False)


    # new microscope doesn't have this?
    #image_loc = output_dir + 'segmented/' + cp.get_CFP(use_id=True)
    #im = Image.open(image_loc)
    #width, height = im.size
    # if height > width:
    #     scale = float(width) / float(height)
    #     x_scaled = (int(scale * cell_size_x))
    #     y_scaled = cell_size_y
    # else:
    #     scale = float(height) / float(width)
    #     x_scaled = cell_size_x
    #     y_scaled = int(scale * cell_size_y)
    # im = im.resize((x_scaled, y_scaled), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(im)
    #CFP_label.configure(image=img)
    #CFP_label.image = img



    # attempt to get distance
    #testimg = cv2.imread(image_loc, cv2.IMREAD_UNCHANGED)
    #image_loc = output_dir + 'segmented/' + cp.get_CFP(use_id=True, outline=False)
    #im = Image.open(image_loc)
    #testimg = np.array(im)

    #gray = cv2.cvtColor(testimg, cv2.COLOR_RGB2GRAY)
    #plt.imshow(gray,  cmap='gray')
    #plt.show()
    #ret, thresh = cv2.threshold(gray, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)

    #contours, h = cv2.findContours(thresh, 1, 2)

    #for cnt in contours:
    #    cv2.drawContours(testimg, [cnt], 0, (0, 0, 255), 1)
    #plt.imshow(testimg)
    #plt.show()





    image_loc = output_dir + 'segmented/' + cp.get_GFP(use_id=True)
    im = im_gfp
    width, height = im.size
    if height > width:
        scale = float(width) / float(height)
        x_scaled = (int(scale * cell_size_x))
        y_scaled = cell_size_y
    else:
        scale = float(height) / float(width)
        x_scaled = cell_size_x
        y_scaled = int(scale * cell_size_y)
    im = im.resize((x_scaled, y_scaled), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im)
    GFP_label.configure(image=img)
    GFP_label.image = img
    GFP_label_text.configure(text="GFP")

    #chk_state = BooleanVar()
    #chk_state.set(False)  # set this with whatever we predict
    #chk = Checkbutton(window, text='Mother-Daughter Pair', var=cp.is_correct)
    #chk.grid(row=6, column=1)
    nuclei_count = 0

    #rad1 = Radiobutton(window, text='One Nuclei', value=1, variable=cp.nuclei_count)
    #rad2 = Radiobutton(window, text='Two Nuclei', value=2, variable=cp.nuclei_count)
    #rad1.grid(row=6, column=2)
    #rad2.grid(row=7, column=2)


    #rad3 = Radiobutton(window, text='One Red Dot', value=1, variable=cp.red_dot_count)
    #rad4 = Radiobutton(window, text='Two Red Dot', value=2, variable=cp.red_dot_count)
    dist_mcherry = customtkinter.CTkLabel(window)
    dist_mcherry.configure(text="Distance: {:.3f}".format(cp.red_dot_distance))
    #rad3.grid(row=6, column=3)
    #rad4.grid(row=7, column=3)
    dist_mcherry.grid(row=7, column=3)

    intensity_mcherry_lbl = customtkinter.CTkLabel(window)
    intensity_mcherry_lbl.configure(text="Line GFP intensity: {}".format(cp.get_mcherry_line_GFP_intensity()))
    intensity_mcherry_lbl.grid(row=8, column=3)


    #rad5 = Radiobutton(window, text='One Cyan Dot', value=1, variable=cp.cyan_dot_count)
    #rad6 = Radiobutton(window, text='Two Cyan Dot', value=2, variable=cp.cyan_dot_count)
    #rad5.grid(row=6, column=5)
    #rad6.grid(row=7, column=5)

    #rad7 = Radiobutton(window, text='One Green Dot', value=1, variable=cp.green_dot_count)
    #rad8 = Radiobutton(window, text='Two Green Dot', value=2, variable=cp.green_dot_count)
    try:
        intense1 = customtkinter.CTkLabel(window)
        intense1.configure(text="Nucleus Intensity Sum: {}".format(cp.nucleus_intensity[Contour.CONTOUR]))
        intense2 = customtkinter.CTkLabel(window)
        intense2.configure(text="Cellular Intensity Sum: {}".format(cp.cell_intensity))
        intense1.grid(row=7, column=4)
        intense2.grid(row=8, column=4)

    except:
        print("error with this cell intensity")
    #rad7.grid(row=6, column=4)
    #rad8.grid(row=7, column=4)
    ignore_txt = 'IGNORE'
    if cp_dict[(image,id)].ignored:
        ignore_txt = 'ENABLE'
    ignore_btn = customtkinter.CTkButton(window, text=ignore_txt, command=partial(ignore, image, id))
    ignore_btn.grid(row=7, column=7, rowspan=2)


    next_btn = customtkinter.CTkButton(window, text="Next Pair", command=partial(display_cell, image, id+1))
    next_btn.grid(row=10, column=4)
    prev_btn = customtkinter.CTkButton(window, text="Previous Pair", command=partial(display_cell, image, id-1))
    prev_btn.grid(row=10, column=2)

    #TODO:  Do this in a less stupid way.
    found_me = False
    next = None
    prev = None
    for k in image_dict.keys():
        if found_me:
            next = k
            break
        if k == image:
            found_me = True
        else:
            prev = k
    if next is None:
        next = list(image_dict.keys())[0]
    if prev is None:
        prev = list(image_dict.keys())[len(image_dict)-1]

    image_next_btn = customtkinter.CTkButton(window, text="Next Image", command=partial(display_cell, next, 1))
    image_next_btn.grid(row=4, column=6)
    image_prev_btn = customtkinter.CTkButton(window, text="Previous Image", command=partial(display_cell, prev, 1))
    image_prev_btn.grid(row=4, column=0)
    cp_dict[(image, id)] = cp
    window.update()


def tink(conf,window1):
    global data
    data = conf
    print(data)
    global window
    window = window1

    window.title("Yeast Analysis Tool")
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    #setting tkinter window size
    window.geometry("%dx%d" % (width, height))
    window.bind("<Configure>", on_resize)

    global drop_ignored
    drop_ignored = BooleanVar()
    drop_ignored.set(True)

    global use_cache
    use_cache = BooleanVar()
    use_cache.set(True)
    if data['useChache'] == 'on':
        use_cache.set(True)
    else:
        use_cache.set(False)

    global use_spc110
    use_spc110 = BooleanVar()
    use_spc110.set(True)

    if data['mCherry_to_find_pairs'] == 'on':
        use_spc110.set(True)
    else:
        use_spc110.set(False)

    global choice_var
    choice_var = data['arrested']


    # choice_var = StringVar(window)
    # choices = ['Metaphase Arrested', 'G1 Arrested']
    # #choice_var.set(list(choices)[0])
    # choice_var.set(choices[0])

    # use_cache_checkbox = Checkbutton(window, text='use cached masks', variable=use_cache)
    # use_cache_checkbox.grid(row=0, column=1)

    # use_mcherry_checkbox = Checkbutton(window, text='use mcherry to find pairs', variable=use_spc110)
    # use_mcherry_checkbox.grid(row=0, column=2)

    # analysis_type_menu = OptionMenu(window, choice_var, choice_var.get(), *choices)   # This seems different than the documentation
    # analysis_type_menu.grid(row=0, column=3)


    global export_btn
    export_btn = customtkinter.CTkButton(window, text='Export to CSV', command= export_to_csv)
    export_btn.grid(row=0, column=4)
    export_btn['state'] = DISABLED

    global drop_ignored_checkbox
    drop_ignored_checkbox = customtkinter.CTkCheckBox(window, text='drop ignored', variable=drop_ignored)
    drop_ignored_checkbox.grid(row=0, column=5)
    drop_ignored_checkbox['state'] = DISABLED

    distvar = StringVar()

    # global input_lbl
    # input_lbl = Label(window, text=input_dir)
    # input_lbl.grid(row=1, column=1, padx=3)

    # global output_lbl
    # output_lbl = Label(window, text=output_dir)
    # output_lbl.grid(row=2, column=1, padx=3)

    global input_dir 
    input_dir= data['input_dir']
    global output_dir 
    output_dir = data['output_dir']

    # input_btn = Button(text="Set Input Directory", command=set_input_directory)
    # input_btn.grid(row=1, column=0)

    # output_btn = Button(text="Set Output Directory", command=set_output_directory)
    # output_btn.grid(row=2, column=0)

    ignore_btn = customtkinter.CTkButton(window, text="IGNORE")
    ignore_btn.grid(row=6, column=7, rowspan=2)

    # kernel_size_lbl = Label(window, text="Kernel Size")
    # kernel_size_lbl.grid(row=1, column=3)

    global kernel_size_input
    kernel_size_input = data['kernel_size']
    # kernel_size_input = Entry(window)
    # kernel_size_input.insert(END, '13')
    # kernel_size_input.grid(row=1, column=4)


    # mcherry_line_width_lbl = Label(window, text="mCherry Line Width")
    # mcherry_line_width_lbl.grid(row=1, column=5)

    global mcherry_line_width_input
    mcherry_line_width_input = data['mCherry_line_width']
    # mcherry_line_width_input = Entry(window)
    # mcherry_line_width_input.insert(END, '1')
    # mcherry_line_width_input.grid(row=1, column=6)


    # kernel_deviation_lbl = Label(window, text="Kernel Deviation")
    # kernel_deviation_lbl.grid(row=2, column=3)

    global kernel_deviation_input
    kernel_deviation_input = data['kernel_diviation']
    # kernel_deviation_input = Entry(window)
    # kernel_deviation_input.insert(END, '5')
    # kernel_deviation_input.grid(row=2, column=4)
    # btn = Button(window, text="Start Analysis", command=segment_images)
    # btn.grid(row=0, column=0)
    global img_title_label
    img_title_label = customtkinter.CTkLabel(window)
    img_title_label.grid(row=3, column=3)

    global img_label
    img_label = customtkinter.CTkLabel(window, text='')
    img_label.grid(row=4, column=1, columnspan=5)

    global ID_label
    ID_label = customtkinter.CTkLabel(window)
    ID_label.grid(row=6, column=0)

    global DIC_label_text
    DIC_label_text = customtkinter.CTkLabel(window, font=("Times New Roman", 18, "bold"))
    DIC_label_text.grid(row=5, column=1)

    global DAPI_label_text
    DAPI_label_text = customtkinter.CTkLabel(window, fg_color='blue', font=("Times New Roman", 18, "bold"))
    DAPI_label_text.grid(row=5, column=2)

    global mCherry_label_text
    mCherry_label_text = customtkinter.CTkLabel(window, fg_color='red', font=("Times New Roman", 18, "bold"))
    mCherry_label_text.grid(row=5, column=3)

    global GFP_label_text
    GFP_label_text = customtkinter.CTkLabel(window, fg_color='green', font=("Times New Roman", 18, "bold"))
    GFP_label_text.grid(row=5, column=4)

    global DIC_label
    DIC_label = customtkinter.CTkLabel(window, text='')
    DIC_label.grid(row=6, column=1)

    global DAPI_label
    DAPI_label = customtkinter.CTkLabel(window, text='')
    DAPI_label.grid(row=6, column=2)

    global mCherry_label
    mCherry_label = customtkinter.CTkLabel(window, text='')
    mCherry_label.grid(row=6, column=3)

    global GFP_label
    GFP_label = customtkinter.CTkLabel(window, text='')
    GFP_label.grid(row=6, column=4)
    img_label.bind("<Button-1>", callback)
    window.bind("<Left>", key)
    window.bind("<Right>", key)
    window.bind("<Up>", key)
    window.bind("<Down>", key)
    for i in range(10):
        window.bind(str(i), key)

    #canvas = Canvas(window, width = 1000, height = 1000)
    segment_images()
    window.mainloop()

#CFP_label = Label(window)
#CFP_label.grid(row=6, column=5)

def callback(event):
    print("clicked at " + str(event.x) + ',' + str(event.y))


keybuf = []

def timedout():
    global current_image
    if keybuf:
        text = ''.join(keybuf)
        keybuf.clear()
        display_cell(current_image, int(text))

def key(event):

    global current_image
    global current_cell


    if current_image == None or current_cell == None:
        return

    if event.char.isdigit():  # lets you type in integers to go to that cell directly
        keybuf.append(event.char)
        window.after(250, timedout)
        return


    #TODO:  Do this in a less stupid way.
    found_me = False
    next = None
    prev = None
    for k in image_dict.keys():
        if found_me:
            next = k
            break
        if k == current_image:
            found_me = True
        else:
            prev = k
    if next is None:
        next = list(image_dict.keys())[0]
    if prev is None:
        prev = list(image_dict.keys())[len(image_dict)-1]
    print ("pressed " + str(repr(event.char)))
    if event.keysym == 'Left':
        display_cell(current_image, current_cell - 1)
    if event.keysym == 'Right':
        display_cell(current_image, current_cell + 1)
    if event.keysym == 'Up':
        display_cell(prev, 0)
    if event.keysym == 'Down':
        display_cell(next, 0)
