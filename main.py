import pytz
from tkinter import *
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

from mrcnn.my_inference import predict_images
from mrcnn.preprocess_images import preprocess_images
from mrcnn.convert_to_image import convert_to_image, convert_to_imagej
from enum import Enum
from cv2_rolling_ball import subtract_background_rolling_ball

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

def set_input_directory():
    global input_dir
    old = input_dir
    global input_lbl
    input_dir = filedialog.askdirectory(parent=window, title='Choose the Directory with the input Images',
                                        initialdir=input_dir)
    #TODO: This updates the variable, but I need to make it update the string on the screen
    if input_dir == "":
        input_dir = old
        return
    input_lbl.config(text = input_dir)
    #print (input_dir)

def set_output_directory():
    global output_dir
    old = output_dir
    global output_lbl
    output_dir = filedialog.askdirectory(parent=window, title='Choose the Directory to output Segmented Images',
                                         initialdir=output_dir)
    if output_dir == "":
        output_dir = old
        return
    output_lbl.config(text = output_dir)

class CellPair:
    def __init__(self, image_name, id):
        self.is_correct = True
        self.image_name = image_name
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
        return self.image_name.split('_R3D_REF')[0]

    def get_DIC(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            return self.get_base_name() + '_R3D_REF' + '-' + str(self.id) + outlinestr + '.tif'
        else:
            return self.get_base_name() + '_R3D_REF.tif'

    def get_DAPI(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w525' + '-' + str(self.id) + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w435' + '-' + str(self.id) + outlinestr + '.tif'
        else:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w525' + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w435' + outlinestr + '.tif'

    def get_GFP(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w625' + '-' + str(self.id)  + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w525' + '-' + str(self.id) + outlinestr + '.tif'
        else:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w625' + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w525' + outlinestr + '.tif'

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
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + 'GFP' + '-' + str(self.id)  + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w625' + '-' + str(self.id) + outlinestr + '.tif'
        else:
            #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + 'GFP' + outlinestr + '.tif'
            return self.get_base_name() + '_PRJ' + '_w625' + outlinestr + '.tif'

    #TODO: Remove is Matt says we will never get this one
    def get_CFP(self, use_id=False, outline=True):
        outlinestr = ''
        if not outline:
            outlinestr = '-no_outline'
        if use_id:
            return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w435' + '-' + str(self.id)  + outlinestr + '.tif'
        else:
            return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w435' + outlinestr + '.tif'

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


    from datetime import datetime
    now = datetime.utcnow().replace(tzinfo=pytz.utc)

    csv_out = filedialog.asksaveasfilename(parent=window, title='Save as...', initialdir='.', defaultextension='.csv') 
    with open(csv_out, mode='w') as outfile:
        outfile_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #write headers
        # Image name, cell id, date, time, software version number, thresholding technique, contour technique, smoothing technique
        outfile_writer.writerow(['imagename', 'cellid',  'datetime', 'kernel size', 'kernel deviation', 'mcherry line width', 'contour type', 'thesholding options', 'nuclear GFP', 'cellular GFP', 'cytoplasmic intensity', 'nuc int/cyto int', 'mcherry distance', 'mcherry line gfp intensity', 'user invalidated'])
        for image, cells  in image_dict.items():
            for cell in cells:
                cp = cp_dict.get((image, cell))
                if cp == None:   # create a new one and run stats
                    cp = CellPair(image, cell) 
                    get_stats(cp)
                    cp_dict[(image, cell)] = cp

                if cp.get_ignored() and drop_ignored.get():
                    continue
                line = list()  #all the elements to write
                try:
                    nucleus_intensity = cp.get_GFP_Nucleus_Intensity(Contour.CONTOUR)[0]
                    cellular_intensity = cp.get_GFP_Cell_Intensity()[0]
                    cytoplasmic_intensity = cellular_intensity - nucleus_intensity
                    nuc_div_cyto_intensity = float(nucleus_intensity)/float(cytoplasmic_intensity)
                except:
                    print('Invalid values in image ' + str(image) + '  and cell ' + str(cell) + '... skipping cell')
                    continue
                line.append(image)
                line.append(cell)
                line.append(now)
                line.append(kernel_size_input.get())
                line.append(kernel_deviation_input.get())
                line.append(mcherry_line_width_input.get())
                line.append('contour')  # we might make this variable in the future
                line.append('cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU') # we might make this variable in the future
                line.append(nucleus_intensity)   # nuclear gfp
                line.append(cellular_intensity)   # cellular gfp
                line.append(cytoplasmic_intensity)
                line.append(nuc_div_cyto_intensity)
                line.append(cp.red_dot_distance)
                line.append(cp.get_mcherry_line_GFP_intensity())
                line.append(cp.get_ignored())   # check if the user has invalidated this sample
                outfile_writer.writerow(line)







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
        if '_R3D_REF' not in image_name:
            continue
        extspl = os.path.splitext(image_name)
        if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
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
            #image_dict[image_name] = segmentation_name
            # Load the original raw image and rescale its intensity values
            image = np.array(Image.open(input_dir + image_name))
            image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            debug_image = image

            # Convert the image to an RGB image, if necessary
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                image = np.expand_dims(image, axis=-1)
                image = np.tile(image, 3)

            # Open the segmentation file    # TODO -- make it show it is choosing the correc segmented
            seg = np.array(Image.open(segmentation_name))   #TODO:  on first run, this can't find outputs/masks/2021_0629_M2210_004_R3D_REF.tif


            #TODO:   If G1 Arrested, we don't want to merge neighbors and ignore non-budding cells
            #choices = ['Metaphase Arrested', 'G1 Arrested']
            outlines = np.zeros(seg.shape)
            if choice_var.get() == 'Metaphase Arrested':
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
                    basename = image_name.split('_R3D_REF')[0]
                    mcherry_dir = input_dir + basename + '_PRJ_TIFFS/'
                    mcherry_image_name = basename + '_PRJ' + '_w625' + '.tif'
                    mcherry_image = np.array(Image.open(mcherry_dir + mcherry_image_name))
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

        base_image_name = image_name.split('_R3D_REF')[0]
        for images in os.listdir(input_dir):
            # don't overlay if it isn't the right base image
            if base_image_name not in images:
                continue
            extspl = os.path.splitext(images)
            if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
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
            image = np.array(Image.open(to_open))
            image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
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

        for full_path in ([filter_dir + x for x in os.listdir(filter_dir)] + [input_dir + image_name,]):
            images = os.path.split(full_path)[1]  # we start in separate directories, but need to end up in the same one
            # don't overlay if it isn't the right base image
            if base_image_name not in images:
                continue
            extspl = os.path.splitext(images)
            if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
                continue
            #tif_image = images.split('.')[0] + '.tif'

            if os.path.isdir(full_path):
                continue
            image = np.array(Image.open(full_path))
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
                cell_tif_image = images.split('.')[0] + '-' + str(i) + '.tif'
                no_outline_image = images.split('.')[0] + '-' + str(i) + '-no_outline.tif'






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
                        csvwriter = csv.writer(csvfile)
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

    if len(image_dict) > 0:
        k, v = list(image_dict.items())[0]
        display_cell(k, v[0])
    #else: show error message

def ignore(image, id):
    global ignore_btn
    global cp_dict
    cp_dict[(image, id)].set_ignored(not cp_dict[(image, id)].get_ignored())
    if cp_dict[(image, id)].get_ignored():
        ignore_btn.config(text = 'ENABLE')
    else:
        ignore_btn.config(text = 'IGNORE')

    # attempt to get distance
    #testimg = cv2.imread(image_loc, cv2.IMREAD_UNCHANGED)





def get_stats(cp):

    global mcherry_line_width_input
    #outlines screw up the analysis
    im = Image.open(output_dir + 'segmented/' + cp.get_mCherry(use_id=True, outline=False))
    im_GFP = Image.open(output_dir + 'segmented/' + cp.get_GFP(use_id=True, outline=False))
    im_GFP_for_cellular_intensity = Image.open(output_dir + 'segmented/' + cp.get_GFP(use_id=True))  #has outline
    testimg = np.array(im)
    GFP_img = np.array(im_GFP)
    img_for_cell_intensity = np.array(im_GFP_for_cellular_intensity)

    cell_intensity_gray = cv2.cvtColor(img_for_cell_intensity, cv2.COLOR_RGB2GRAY)

    # was RGBA2GRAY
    orig_gray_GFP = cv2.cvtColor(GFP_img, cv2.COLOR_RGB2GRAY)
    orig_gray_GFP_no_bg, background = subtract_background_rolling_ball(orig_gray_GFP, 50, light_background=False,
                                                       use_paraboloid=False, do_presmooth=True)
    orig_gray = cv2.cvtColor(testimg, cv2.COLOR_RGB2GRAY)
    kdev = int(kernel_deviation_input.get())
    ksize = int(kernel_size_input.get())
    #ksize must be odd
    if ksize%2 == 0:
        ksize += 1
        print("You used an even ksize, updating to odd number +1")
    gray_mcherry=cv2.GaussianBlur(orig_gray, (3,3), 1)
    ret_mcherry, thresh_mcherry = cv2.threshold(gray_mcherry, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
    gray = cv2.GaussianBlur(orig_gray, (ksize,ksize), kdev)
    # plt.title("blur")
    # plt.imshow(gray,  cmap='gray')
    # plt.show()
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)


    # Some of the cell outlines are split into two circles.  We blur this so that the contour will cover  both of them

    cell_intensity_gray = cv2.GaussianBlur(cell_intensity_gray, (3,3), 1)
    # plt.title("cell_int_gray")
    # plt.imshow(cell_intensity_gray, cmap='gray')
    # plt.show()
    # plt.title("cell_int")
    # plt.imshow(img_for_cell_intensity, cmap='gray')
    # plt.show()
    cell_int_ret, cell_int_thresh = cv2.threshold(cell_intensity_gray, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
    cell_int_cont, cell_int_h = cv2.findContours(cell_int_thresh, 1, 2)

    #we want the biggest contour because that'll be our cellular intensity boundary
    largest = 0
    largest_cell_cnt = None
    #TODO:  I think I should join the two largest that aren't the full box
    for i, cnt in enumerate(cell_int_cont):
        #if i == len(cell_int_cont) - 1:    # this is not robust #TODO fix it  -- this throws out the full box contour
        #    continue
        area = cv2.contourArea(cnt)
        if  area > largest:
            largest = area
            largest_cell_cnt = cnt


    contours, h = cv2.findContours(thresh, 1, 2)
    contours_mcherry = cv2.findContours(thresh_mcherry, 1, 2)
    #iterate through contours and throw out the largest (the box) and anything less than the second and third largest)
    # Contours finds the entire image as a contour and it seems to always put it in the contours[len(contours)].  We should do this more robustly in the future



    #bestContours, bestArea = find_best_contours(contours)
    #bestContours_mcherry, bestArea_mcherry = find_best_contours(contours_mcherry)

    #these include the outlines already, so lets edit them
    edit_im = Image.open(output_dir + 'segmented/' + cp.get_mCherry(use_id=True))
    edit_im_GFP = Image.open(output_dir + 'segmented/' + cp.get_GFP(use_id=True))
    edit_testimg = np.array(edit_im)
    edit_GFP_img = np.array(edit_im_GFP)
    #edit_testimg = cv2.cvtColor(edit_testimg, cv2.COLOR_GRAY2BGR)
    #edit_GFP_img = cv2.cvtColor(edit_GFP_img, cv2.COLOR_GRAY2BGR)
    best_contour = None

    bestContours = list()
    bestArea = list()
    for i, cnt in enumerate(contours):
        #tester = orig_gray
        if len(cnt) == 0:
            continue
        #cv2.drawContours(tester, cnt, 0, 255, 1)
        #plt.imshow(tester,  cmap='gray')
        #plt.show()
        if i == len(contours) - 1:    # this is not robust #TODO fix it
            continue
        area = cv2.contourArea(cnt)

        if len(bestContours) == 0:
            bestContours.append(i)
            bestArea.append(area)
            continue
        if len(bestContours) == 1:
            bestContours.append(i)
            bestArea.append(area)
        if area > bestArea[0]:
            bestArea[1] = bestArea[0]
            bestArea[0] = area
            bestContours[1] = bestContours[0]
            bestContours[0] = i
        elif area > bestArea[1]:    # probably won't have a 3rd that is equal, but that would cause a problem
            bestArea[1] = area
            bestContours[1] = i


    if len(bestContours) == 0:
        print("we didn't find any contours")
        return edit_im, edit_im_GFP

    bestContours_mcherry = list()
    bestArea_mcherry = list()
    for i, cnt in enumerate(contours_mcherry[0]):
        #tester = orig_gray
        if len(cnt) == 0:
            continue
        #cv2.drawContours(tester, cnt, 0, 255, 1)
        #plt.imshow(tester,  cmap='gray')
        #plt.show()
        if i == len(contours_mcherry[0]) - 1:    # this is not robust #TODO fix it
            continue
        try:
            area = cv2.contourArea(cnt)
        except:  # no area
            continue

        if len(bestContours_mcherry) == 0:
            bestContours_mcherry.append(i)
            bestArea_mcherry.append(area)
            continue
        if len(bestContours_mcherry) == 1:
            bestContours_mcherry.append(i)
            bestArea_mcherry.append(area)
        if area > bestArea_mcherry[0]:
            bestArea_mcherry[1] = bestArea_mcherry[0]
            bestArea_mcherry[0] = area
            bestContours_mcherry[1] = bestContours_mcherry[0]
            bestContours_mcherry[0] = i
        elif area > bestArea_mcherry[1]:    # probably won't have a 3rd that is equal, but that would cause a problem
            bestArea_mcherry[1] = area
            bestContours_mcherry[1] = i

    mcherry_line_pts = list()
    if len(bestContours_mcherry) == 2:
        c1 = contours_mcherry[0][bestContours_mcherry[0]]
        c2 = contours_mcherry[0][bestContours_mcherry[1]]
        M1 = cv2.moments(c1)
        M2 = cv2.moments(c2)
        # TODO:  This was code from when we wanted to get the distance between the mCherry centers
        if M1['m00'] == 0 or M2['m00'] == 0:   # something has gone wrong
            print("Warning:  The m00 moment = 0")
            #plt.imshow(edit_testimg,  cmap='gray')
            #plt.show()
        else:

            c1x = int(M1['m10'] / M1['m00'])
            c1y = int(M1['m01'] / M1['m00'])
            c2x = int(M2['m10'] / M2['m00'])
            c2y = int(M2['m01'] / M2['m00'])
            d = math.sqrt(pow(c1x - c2x, 2) + pow(c1y - c2y, 2))
            #print ('Distance: ' + str(d))
            cp.set_red_dot_distance(d)
            cv2.line(edit_testimg, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input.get()))
            mcherry_line_mask = np.zeros(gray.shape, np.uint8)
            cv2.line(mcherry_line_mask, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input.get()))
            mcherry_line_pts = np.transpose(np.nonzero(mcherry_line_mask))

    if len(bestContours) == 2:  # "There can be only one!" - Connor MacLeod
        c1 = contours[bestContours[0]]
        c2 = contours[bestContours[1]]
        MERGE_CLOSEST = True
        if MERGE_CLOSEST:   # find the two closest points and just push c2 into c1 there
            smallest_distance = 999999999
            second_smallest_distance = 999999999
            smallest_pair = (-1, -1)  # invalid so it'll cause an error if it is used
            for pt1 in c1:
                for i, pt2 in enumerate(c2):
                    d = math.sqrt(pow(pt1[0][0] - pt2[0][0], 2) + pow(pt1[0][1] - pt2[0][1], 2))
                    if d < smallest_distance:
                        second_smallest_distance = smallest_distance
                        second_smallest_pair = smallest_pair
                        smallest_distance = d
                        smallest_pair = (pt1, pt2, i)
                    elif d < second_smallest_distance:
                        second_smallest_distance = d
                        second_smallest_pair = (pt1, pt2, i)

            #now we have the two closest points to each other between the two contours
            # iterate through the contour 1 until you find pt1, then add every point in c2
            # to c1, starting from pt2 until you reach pt2 of the second_smallest pair, then
            # remove points in c1 until you reach pt1 of second_smallest_pair
            clockwise = True  # we need to figure out which  of the two points shoudl go first

            best_contour = list()
            #temp hacky way of doing it
            for pt1 in c1:
                best_contour.append(pt1)
                if pt1[0].tolist() != smallest_pair[0][0].tolist():
                    continue
                # we are at the closest p1
                start_loc = smallest_pair[2]
                finish_loc = start_loc - 1
                if start_loc == 0:
                    finish_loc = len(c2) - 1
                current_loc = start_loc
                while current_loc != finish_loc:
                    best_contour.append(c2[current_loc])
                    current_loc += 1
                    if current_loc >= len(c2):
                        current_loc = 0
                #grab the last point
                best_contour.append(c2[finish_loc])
            best_contour = np.array(best_contour).reshape((-1, 1, 2)).astype(np.int32)







# After recent edits this should always be true, but leaving it here in case I messed up.
    if len(bestContours) == 1:
        best_contour = contours[bestContours[0]]


    print("only 1 contour found")
    #M1 = cv2.moments(best_contour)
    #(x1, y1), radius1 = cv2.minEnclosingCircle(best_contour)
    #center1 = (int(x1), int(y1))
    #radius1 = int(radius1)
    #h1 = cv2.convexHull(best_contour)
    #cv2.drawContours(edit_testimg, [h1, ], 0, 255, 1)
    #cv2.drawContours(edit_GFP_img, [h1, ], 0, 255, 1)
    cv2.drawContours(edit_testimg, [best_contour], 0, (0, 255, 0), 1)
    cv2.drawContours(edit_GFP_img, [best_contour], 0, (0, 255, 0), 1)
    #cv2.drawContours(edit_GFP_img, [largest_cell_cnt], 0, (255,0,0), 1)

    # Circles instead of contours
    #cv2.circle(edit_testimg, center1, radius1, (255, 0, 0), 1)
    #cv2.circle(edit_GFP_img, center1, radius1, (255, 0, 0), 1)

    # compute intensities
    # lets edit the outlined images, not the regulars
    #mask_circle = np.zeros(gray.shape, np.uint8)
    #mask_convex = np.zeros(gray.shape, np.uint8)
    mask_contour = np.zeros(gray.shape, np.uint8)
    cell_mask = np.zeros(gray.shape, np.uint8)

    # actual contour mask?
    #        cv2.drawContours(mask, [contours[cnt]], 255, 100, -1)

    # Circle Mask
    #cv2.circle(mask_circle, center1, radius1, 255, -1)
    #cv2.drawContours(mask_convex, [h1, ], 0, 255, 1)
    #cv2.drawContours(mask_contour, [best_contour], 0, 255, 1)
    cv2.fillPoly(mask_contour, [best_contour], 255)


    # read in the outline file if you need it
    border_cells = []
    with open(output_dir + 'masks/' + cp.get_base_name() + '-' + str(cp.id) + '.outline', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            border_cells.append([int(row[0]), int(row[1])])

    #cell_list = np.array(border_cells).reshape((-1, 1, 2)).astype(np.int32)
    #cv2.fillPoly(cell_mask, [cell_ctr], 255)

    #approx = cv2.approxPolyDP(largest_cell_cnt, 0.01 * cv2.arcLength(largest_cell_cnt, True), True)
    #cv2.drawContours(cell_mask, [largest_cell_cnt], 0, 255, 1)
    #cv2.fillPoly(cell_mask, [largest_cell_cnt], 255)

    #pts_circle = np.transpose(np.nonzero(mask_circle))
    #pts_convex = np.transpose(np.nonzero(mask_convex))
    pts_contour = np.transpose(np.nonzero(mask_contour))

    #cell_pts_contour = np.transpose(np.nonzero(cell_mask))

    # intensity_sum = 0
    # for p in pts_circle:
    #     intensity_sum += orig_gray[p[0]][p[1]]
    # cp.set_GFP_Nucleus_Intensity(Contour.CIRCLE, intensity_sum, len(pts_circle))

    # intensity_sum = 0
    # for p in pts_convex:
    #     intensity_sum += orig_gray[p[0]][p[1]]
    # cp.set_GFP_Nucleus_Intensity(Contour.CONTOURS, intensity_sum, len(pts_convex))

    intensity_sum = 0
    for p in pts_contour:
        intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]
    cp.set_GFP_Nucleus_Intensity(Contour.CONTOUR, intensity_sum, len(pts_contour))



    cell_intensity_sum = 0
    for p in border_cells:
        cell_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]
    cp.set_GFP_Cell_Intensity(cell_intensity_sum, len(border_cells))


    mcherry_line_intensity_sum = 0

    for p in mcherry_line_pts:
        mcherry_line_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]
    cp.set_mcherry_line_GFP_intensity(mcherry_line_intensity_sum)





    return Image.fromarray(edit_testimg), Image.fromarray(edit_GFP_img)

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
    cp = cp_dict.get((image, id))
    if cp == None:
        cp = CellPair(image, id)
        cp_dict[(image, id)] = cp
    main_size_x = int(0.5 * win_width)
    main_size_y = int(0.5 * win_height)
    cell_size_x = int(0.23*main_size_x)
    cell_size_y = int(0.23*main_size_x)

    im_cherry, im_gfp = get_stats(cp)
    image_loc = output_dir + 'segmented/' + cp.get_DIC()  #TODO:  This messes up with the _w50 naming
    im = Image.open(image_loc)
    width, height = im.size
    if height > width:
        scale = float(width)/float(height)
    else:
        scale = float(height) / float(width)
    im = im.resize((int(scale * main_size_x), main_size_y), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im)
    img_label.configure(image=img)
    img_label.image = img


    image_loc = output_dir + 'segmented/' + cp.get_DIC(use_id=True)
    im = Image.open(image_loc)
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
    DIC_label.configure(image=img)
    DIC_label.image = img
    DIC_label_text.configure(text="DIC")

    image_loc = output_dir + 'segmented/' + cp.get_DAPI(use_id=True)
    im = Image.open(image_loc)
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
    dist_mcherry = Label(window)
    dist_mcherry.config(text="Distance: {:.3f}".format(cp.red_dot_distance))
    #rad3.grid(row=6, column=3)
    #rad4.grid(row=7, column=3)
    dist_mcherry.grid(row=7, column=3)

    intensity_mcherry_lbl = Label(window)
    intensity_mcherry_lbl.config(text="Line GFP intensity: {}".format(cp.get_mcherry_line_GFP_intensity()))
    intensity_mcherry_lbl.grid(row=8, column=3)


    #rad5 = Radiobutton(window, text='One Cyan Dot', value=1, variable=cp.cyan_dot_count)
    #rad6 = Radiobutton(window, text='Two Cyan Dot', value=2, variable=cp.cyan_dot_count)
    #rad5.grid(row=6, column=5)
    #rad6.grid(row=7, column=5)

    #rad7 = Radiobutton(window, text='One Green Dot', value=1, variable=cp.green_dot_count)
    #rad8 = Radiobutton(window, text='Two Green Dot', value=2, variable=cp.green_dot_count)
    try:
        intense1 = Label(window)
        intense1.config(text="Nucleus Intensity Sum: {}".format(cp.nucleus_intensity[Contour.CONTOUR]))
        intense2 = Label(window)
        intense2.config(text="Cellular Intensity Sum: {}".format(cp.cell_intensity))
        intense1.grid(row=7, column=4)
        intense2.grid(row=8, column=4)

    except:
        print("error with this cell intensity")
    #rad7.grid(row=6, column=4)
    #rad8.grid(row=7, column=4)
    ignore_txt = 'IGNORE'
    if cp_dict[(image,id)].ignored:
        ignore_txt = 'ENABLE'
    ignore_btn = Button(window, text=ignore_txt, command=partial(ignore, image, id))
    ignore_btn.grid(row=7, column=7, rowspan=2)


    next_btn = Button(window, text="Next Pair", command=partial(display_cell, image, id+1))
    next_btn.grid(row=10, column=4)
    prev_btn = Button(window, text="Previous Pair", command=partial(display_cell, image, id-1))
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

    image_next_btn = Button(window, text="Next Image", command=partial(display_cell, next, 1))
    image_next_btn.grid(row=4, column=6)
    image_prev_btn = Button(window, text="Previous Image", command=partial(display_cell, prev, 1))
    image_prev_btn.grid(row=4, column=0)
    cp_dict[(image, id)] = cp
    window.update()

window = Tk()

window.title("Yeast Analysis Tool")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
#setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.bind("<Configure>", on_resize)
btn = Button(window, text="Start Analysis", command=segment_images)
btn.grid(row=0, column=0)


drop_ignored = BooleanVar()
drop_ignored.set(True)

use_cache = BooleanVar()
use_cache.set(True)

use_spc110 = BooleanVar()
use_spc110.set(True)

choice_var = StringVar(window)
choices = ['Metaphase Arrested', 'G1 Arrested']
#choice_var.set(list(choices)[0])
choice_var.set(choices[0])

use_cache_checkbox = Checkbutton(window, text='use cached masks', variable=use_cache)
use_cache_checkbox.grid(row=0, column=1)

use_mcherry_checkbox = Checkbutton(window, text='use mcherry to find pairs', variable=use_spc110)
use_mcherry_checkbox.grid(row=0, column=2)

analysis_type_menu = OptionMenu(window, choice_var, choice_var.get(), *choices)   # This seems different than the documentation
analysis_type_menu.grid(row=0, column=3)



export_btn = Button(window, text='Export to CSV', command=export_to_csv)
export_btn.grid(row=0, column=4)
export_btn['state'] = DISABLED


drop_ignored_checkbox = Checkbutton(window, text='drop ignored', variable=drop_ignored)
drop_ignored_checkbox.grid(row=0, column=5)
drop_ignored_checkbox['state'] = DISABLED

distvar = StringVar()


input_lbl = Label(window, text=input_dir)
input_lbl.grid(row=1, column=1, padx=3)

output_lbl = Label(window, text=output_dir)
output_lbl.grid(row=2, column=1, padx=3)

input_btn = Button(text="Set Input Directory", command=set_input_directory)
input_btn.grid(row=1, column=0)

output_btn = Button(text="Set Output Directory", command=set_output_directory)
output_btn.grid(row=2, column=0)

ignore_btn = Button(window, text="IGNORE")
#ignore_btn.grid(row=6, column=7, rowspan=2)

kernel_size_lbl = Label(window, text="Kernel Size")
kernel_size_lbl.grid(row=1, column=3)

kernel_size_input = Entry(window)
kernel_size_input.insert(END, '13')
kernel_size_input.grid(row=1, column=4)


mcherry_line_width_lbl = Label(window, text="mCherry Line Width")
mcherry_line_width_lbl.grid(row=1, column=5)

mcherry_line_width_input = Entry(window)
mcherry_line_width_input.insert(END, '1')
mcherry_line_width_input.grid(row=1, column=6)


kernel_deviation_lbl = Label(window, text="Kernel Deviation")
kernel_deviation_lbl.grid(row=2, column=3)


kernel_deviation_input = Entry(window)
kernel_deviation_input.insert(END, '5')
kernel_deviation_input.grid(row=2, column=4)

img_title_label = Label(window)
img_title_label.grid(row=3, column=3)

img_label = Label(window)
img_label.grid(row=4, column=1, columnspan=5)

ID_label = Label(window)
ID_label.grid(row=6, column=0)

DIC_label_text = Label(window, font=("Times New Roman", 18, "bold"))
DIC_label_text.grid(row=5, column=1)

DAPI_label_text = Label(window, foreground='blue', font=("Times New Roman", 18, "bold"))
DAPI_label_text.grid(row=5, column=2)

mCherry_label_text = Label(window, foreground='red', font=("Times New Roman", 18, "bold"))
mCherry_label_text.grid(row=5, column=3)

GFP_label_text = Label(window, foreground='green', font=("Times New Roman", 18, "bold"))
GFP_label_text.grid(row=5, column=4)


DIC_label = Label(window)
DIC_label.grid(row=6, column=1)

DAPI_label = Label(window)
DAPI_label.grid(row=6, column=2)

mCherry_label = Label(window)
mCherry_label.grid(row=6, column=3)

GFP_label = Label(window)
GFP_label.grid(row=6, column=4)

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

img_label.bind("<Button-1>", callback)
window.bind("<Left>", key)
window.bind("<Right>", key)
window.bind("<Up>", key)
window.bind("<Down>", key)
for i in range(10):
    window.bind(str(i), key)

#canvas = Canvas(window, width = 1000, height = 1000)




window.mainloop()
