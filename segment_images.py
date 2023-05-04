from scipy.spatial import distance as dist
import math

import opts as opt
import os
import shutil

import csv
import cv2
import numpy as np
from PIL import Image
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
import services
from export import export_to_csv_file

global data
def get_neighbor_count(seg_image, center, radius=1):
        # TODO:  account for loss as distance gets larger
        neighbor_list = []
        neighbors = seg_image[center[0] - radius:center[0] + radius + 1, center[1] - radius:center[1] + radius + 1]
        for x, row in enumerate(neighbors):
                neighbor_list.extend(
                    val for y, val in enumerate(row)
                    if (x, y) != (radius, radius) and int(val) != 0
                    and int(val) != int(seg_image[center[0], center[1]]))
        return neighbor_list

def process_images(use_cache):
    preprocessed_image_directory = f"{output_dir}preprocessed_images/"
    preprocessed_image_list = f"{output_dir}preprocessed_images_list.csv"
    rle_file = f"{output_dir}compressed_masks.csv"
    output_mask_directory = f"{output_dir}masks/"
    output_imagej_directory = f"{output_dir}imagej/"

    # Preprocess the images
    if opt.verbose:
        print("\nPreprocessing your images...")
    # TODO:  put everything in separate directories -- list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    preprocess_images(input_dir,
                        output_mask_directory,
                        preprocessed_image_directory,
                        preprocessed_image_list,
                        verbose=opt.verbose,
                        use_cache=use_cache)

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

def segment_images(conf, use_cache, use_spc110):
    global image_dict
    # TODO:  ask user if they want to refresh segmentation
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

    data = conf
    print(data)
    global input_dir
    input_dir = data['input_dir']
    global output_dir
    output_dir = data['output_dir']
    kernel_size_input = data['kernel_size']
    global mcherry_line_width_input
    mcherry_line_width_input = data['mCherry_line_width']
    kernel_deviation_input = data['kernel_diviation']
    choice_var = data['arrested']
    img_format = data['img_format']
    cp_dict = {}
    image_dict = {}

    if input_dir[-1] != "/":
        input_dir = f"{input_dir}/"
    if output_dir[-1] != "/":
        output_dir = f"{output_dir}/"

    if output_dir != '' and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.isdir(output_dir):
        process_images(use_cache)

    for image_name in os.listdir(input_dir):
        if img_format == 'tiff':
            if '_R3D_REF' not in image_name:
                continue
            extspl = os.path.splitext(image_name)
            if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
                continue
        else:
            if '_PRJ' not in image_name:
                continue
            extspl = os.path.splitext(image_name)
            if len(extspl) != 2 or extspl[1] != '.dv':  # ignore files that aren't dv
                continue
        seg = None
        image_dict[image_name] = []
        # cp = CellPair(output_dir + 'segmented/' + image_name.split('.')[0] + '.tif', output_dir + 'masks/' + image_name.split('.')[0] + '.tif')
        exist_check = os.path.exists(
            f'{output_dir}masks/'
            + image_name.split('.')[0]
            + '-cellpairs'
            + '.tif'
        )
        if exist_check and use_cache:
            seg = np.array(
                Image.open(
                    f'{output_dir}masks/'
                    + image_name.split('.')[0]
                    + '-cellpairs'
                    + '.tif'
                )
            )
                    # outlines = np.zeros(seg.shape)
        else:

            segmentation_name = f'{output_dir}masks/{image_name}'
            # image_dict[image_name] = segmentation_name
            # Load the original raw image and rescale its intensity values
            if img_format == 'tiff':
                image = np.array(Image.open(input_dir + image_name))
                image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
            else:
                seg_ext = os.path.splitext(segmentation_name)
                segmentation_name = seg_ext[0] + '.tif'
                f = DVFile(input_dir + image_name)
                im = f.asarray()
                image = Image.fromarray(im[0])
                image = skimage.exposure.rescale_intensity(np.float32(image), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            debug_image = image

            # Convert the image to an RGB image, if necessary
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.expand_dims(image, axis=-1)
                image = np.tile(image, 3)

            # Open the segmentation file    # TODO -- make it show it is choosing the correct segmented
            seg = np.array(Image.open(segmentation_name))   #TODO:  on first run, this can't find outputs/masks/M***.tif'

            # TODO:   If G1 Arrested, we don't want to merge neighbors and ignore non-budding cells
            # choices = ['Metaphase Arrested', 'G1 Arrested']
            outlines = np.zeros(seg.shape)
            lines_to_draw = {}
            if choice_var == 'Metaphase Arrested':
                # Create a raw file to store the outlines

                ignore_list = []
                single_cell_list = []
                # merge cell pairs
                neighbor_count = {}
                closest_neighbors = {}
                for i in range(1, int(np.max(seg) + 1)):
                    cells = np.where(seg == i)
                    # examine neighbors
                    neighbor_list = []
                    for cell in zip(cells[0], cells[1]):
                        # TODO:  account for going over the edge without throwing out the data

                        try:
                            neighbor_list = get_neighbor_count(seg, cell, 3)
                        except Exception:
                            continue

                        for neighbor in neighbor_list:
                            if int(neighbor) in [i, 0]:
                                continue
                            if neighbor in neighbor_count:
                                neighbor_count[neighbor] += 1
                            else:
                                neighbor_count[neighbor] = 1

                    sorted_dict = dict(sorted(neighbor_count.items(), key=lambda item: item[1]))
                    # v = list(neighbor_count.values())
                    # k = list(neighbor_count.keys())
                    if not sorted_dict:
                        print(f'found single cell at: {str(cell)}')
                        single_cell_list.append(int(i))
                    else:
                        # closest_neighbor = k[v.index(max(v))]
                        if len(sorted_dict) == 1:
                            closest_neighbors[i] = list(sorted_dict.items())[0][0]
                        else:
                            top_val = list(sorted_dict.items())[0][1]
                            second_val = list(sorted_dict.items())[1][1]
                            if second_val > 0.5 * top_val:  # things got confusing, so we throw it and its neighbor out
                                single_cell_list.append(int(i))
                                for cluster_cell in neighbor_count:
                                    single_cell_list.append(int(cluster_cell))
                            else:
                                closest_neighbors[i] = list(sorted_dict.items())[0][0]

                    # reset for the next cell
                    neighbor_count = {}
                # TODO:  Examine the spc110 dots and make closest dots neighbors
                resolve_cells_using_spc110 = use_spc110
                if resolve_cells_using_spc110:

                    # open the mcherry
                    if img_format == 'tiff':
                        basename = image_name.split('_R3D_REF')[0]
                        mcherry_dir = input_dir + basename + '_PRJ_TIFFS/'
                        mcherry_image_name = f'{basename}_PRJ' + '_w625' + '.tif'
                        mcherry_image = np.array(Image.open(mcherry_dir + mcherry_image_name))
                    else:
                        f = DVFile(input_dir + image_name)
                        mcherry_image = f.asarray()[3]

                    mcherry_image = skimage.exposure.rescale_intensity(mcherry_image.astype(np.float32),
                                                                       out_range=(0, 1))
                    mcherry_image = np.round(mcherry_image * 255).astype(np.uint8)

                    # Convert the image to an RGB image, if necessary
                    if (
                        len(mcherry_image.shape) != 3
                        or mcherry_image.shape[2] != 3
                    ):
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

                    # mcherry_image_gray = cv2.GaussianBlur(mcherry_image_gray, (1, 1), 0)
                    mcherry_image_ret, mcherry_image_thresh = cv2.threshold(mcherry_image_gray, 0, 1,
                                                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
                    mcherry_image_cont, mcherry_image_h = cv2.findContours(mcherry_image_thresh, 1, 2)

                    if debug:
                        cv2.drawContours(image, mcherry_image_cont, -1, 255, 1)
                        plt.figure(dpi=600)
                        plt.title("ref image with contours")
                        plt.imshow(image, cmap='gray')
                        plt.show()

                    # 921,800

                    min_mcherry_distance = {}
                    min_mcherry_loc = {}
                    for cnt1 in mcherry_image_cont:
                        try:
                            contourArea = cv2.contourArea(cnt1)
                            if contourArea > 100000:  # test for the big box, TODO: fix this to be adaptive
                                print('threw out the bounding box for the entire image')
                                continue
                            M1 = cv2.moments(cnt1)
                            # These are opposite of what we would expect
                            c1x, c1y = services.getMoments(M1)


                        except Exception:
                            continue
                        c_id = int(seg[c1x][c1y])
                        if c_id == 0:
                            continue
                        for cnt2 in mcherry_image_cont:
                            try:
                                M2 = cv2.moments(cnt2)
                                # find center of each contour
                                c2x, c2y = services.getMoments(M2)



                            except Exception:
                                continue  # no moment found
                            if int(seg[c2x][c2y]) == 0:
                                continue
                            if seg[c1x][c1y] == seg[c2x][
                                c2y]:  # these are ihe same cell already -- Maybe this is ok?  TODO:  Figure out hwo to handle this because some of the mcherry signals are in the same cell
                                continue
                            # find the closest point to each center
                            d = math.sqrt(pow(c1x - c2x, 2) + pow(c1y - c2y, 2))
                            if min_mcherry_distance.get(c_id) is None:
                                min_mcherry_distance[c_id] = d
                                min_mcherry_loc[c_id] = int(seg[c2x][c2y])
                                lines_to_draw[c_id] = ((c1y, c1x), (c2y, c2x))
                            else:
                                if d < min_mcherry_distance[c_id]:
                                    min_mcherry_distance[c_id] = d
                                    min_mcherry_loc[c_id] = int(seg[c2x][c2y])
                                    lines_to_draw[c_id] = ((c1y, c1x), (c2y, c2x))  # flip it back here
                                elif d == min_mcherry_distance[c_id]:
                                    print(
                                        'This is unexpected, we had two mcherry red dots in cells {} and {} at the same distance from ('.format(
                                            seg[c1x][c1y], seg[c2x][c2y]) + str(min_mcherry_loc[c_id]) + ', ' + str(
                                            (c2x, c2y)) + ') to ' + str((c1x, c1y)) + ' at a distance of ' + str(d))

                for k, v in closest_neighbors.items():
                    if v in closest_neighbors:  # check to see if v could be a mutual pair
                        if int(v) in ignore_list:  # if we have already paired this one, throw it out
                            single_cell_list.append(int(k))
                            continue

                        if closest_neighbors[int(v)] == int(k) and int(
                                k) not in ignore_list:  # closest neighbors are reciprocal
                            # TODO:  set them to all be the same cell
                            to_update = np.where(seg == v)
                            ignore_list.append(int(v))
                            if resolve_cells_using_spc110:
                                if int(v) in min_mcherry_loc:  # if we merge them here, we don't need to do it with mcherry
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
                        if int(c_id) in ignore_list:  # if we have already paired this one, ignore it
                            continue
                        if int(nearest_cid) in min_mcherry_loc:  # make sure teh reciprocal exists
                            if min_mcherry_loc[int(nearest_cid)] == int(c_id) and int(
                                    c_id) not in ignore_list:  # if it is mutual
                                print('added a cell pair in image {} using the mcherry technique {} and {}'.format(
                                    image_name, int(nearest_cid),
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
                                print(
                                    'could not add cell pair because cell {} and cell {} were not mutually closest'.format(
                                        nearest_cid, int(v)))
                                single_cell_list.append(int(k))

                # remove single cells or confusing cells
                for cell in single_cell_list:
                    seg[np.where(seg == cell)] = 0.0

                # only merge if two cells are both each others closest neighbors
                # otherwise zero them out?
                # rebase segment count
                to_rebase = []
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
                seg_image.save(output_dir + 'masks/' + image_name.split('.')[0] + '-cellpairs' + '.tif')
        for i in range(1, int(np.max(seg)) + 1):
            image_dict[image_name].append(i)

        if img_format == 'tiff':
            base_image_name = image_name.split('_R3D_REF')[0]
        else:
            base_image_name = image_name.split('_PRJ')[0]
        for images in os.listdir(input_dir):
            # don't overlay if it isn't the right base image
            if base_image_name not in images:
                continue
            extspl = os.path.splitext(images)
            if img_format == 'tiff':
                if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
                    continue
                if_g1 = ''
                # if choice_var.get() == 'G1 Arrested':   #if it is a g1 cell, do we really need a separate type of file?
                #    if_g1 = '-g1'
                tif_image = images.split('.')[0] + if_g1 + '.tif'
                if os.path.exists(output_dir + 'segmented/' + tif_image) and use_cache:
                    continue
                to_open = input_dir + images
                if os.path.isdir(to_open):
                    continue
                image = np.array(Image.open(to_open))
                image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
            else:
                if len(extspl) != 2 or extspl[1] != '.dv':  # ignore files that aren't dv
                    continue
                if_g1 = ''
                tif_image = images.split('.')[0] + if_g1 + '.tif'
                if os.path.exists(output_dir + 'segmented/' + tif_image) and use_cache:
                    continue
                to_open = input_dir + images
                if os.path.isdir(to_open):
                    continue
                f = DVFile(to_open)
                im = f.asarray()
                image = Image.fromarray(im[0])
                image = skimage.exposure.rescale_intensity(np.float32(image), out_range=(0, 1))
            image = np.round(image * 255).astype(np.uint8)

            # Convert the image to an RGB image, if necessary
            if len(image.shape) != 3 or image.shape[2] != 3:
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
                cv2.line(image_outlined, start, stop, (255, 0, 0), 1)

            # iterate over each cell pair and add an ID to the image
            for i in range(1, int(np.max(seg) + 1)):
                loc = np.where(seg == i)
                if len(loc[0]) > 0:
                    txt = ax.text(loc[1][0], loc[0][0], str(i), size=12)
                    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
                else:
                    print('could not find cell id ' + str(i))

            fig.savefig(output_dir + 'segmented/' + tif_image, dpi=600, bbox_inches='tight', pad_inches=0)

                    # plt.show()

        # TODO:  Combine the two iterations over the input directory images

        # This is where we overlay what we learned in the DIC onto the other images

        filter_dir = input_dir + base_image_name + '_PRJ_TIFFS/'
        if img_format == 'tiff':
            for full_path in ([filter_dir + x for x in os.listdir(filter_dir)] + [input_dir + image_name, ]):
                images = os.path.split(full_path)[1]  # we start in separate directories, but need to end up in the same one
                # don't overlay if it isn't the right base image
                if base_image_name not in images:
                    continue
                extspl = os.path.splitext(images)
                if len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
                    continue
                # tif_image = images.split('.')[0] + '.tif'

                if os.path.isdir(full_path):
                    continue
                image = np.array(Image.open(full_path))
                image = skimage.exposure.rescale_intensity(image.astype(np.float32), out_range=(0, 1))
                image = np.round(image * 255).astype(np.uint8)

                # Convert the image to an RGB image, if necessary
                if len(image.shape) != 3 or image.shape[2] != 3:
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

                    a = np.where(seg == i)  # somethin bad is happening when i = 4 on my tests
                    min_x = max(np.min(a[0]) - 1, 0)
                    max_x = min(np.max(a[0]) + 1, seg.shape[0])
                    min_y = max(np.min(a[1]) - 1, 0)
                    max_y = min(np.max(a[1]) + 1, seg.shape[1])

                    # a[0] contains the x coords and a[1] contains the y coords
                    # save this to use later when I want to calculate cellular intensity

                    # convert from absolute location to relative location for later use

                    if not os.path.exists(
                            output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline') or not use_cache:
                        with open(output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline', 'w') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerows(zip(a[0] - min_x, a[1] - min_y))

                    cellpair_image = image_outlined[min_x: max_x, min_y:max_y]
                    not_outlined_image = image[min_x: max_x, min_y:max_y]
                    if not os.path.exists(
                            output_dir + 'segmented/' + cell_tif_image) or not use_cache:  # don't redo things we already have
                        plt.imsave(output_dir + 'segmented/' + cell_tif_image, cellpair_image, dpi=600, format='TIFF')
                        plt.clf()
                    if not os.path.exists(
                            output_dir + 'segmented/' + no_outline_image) or not use_cache:  # don't redo things we already have
                        plt.imsave(output_dir + 'segmented/' + no_outline_image, not_outlined_image, dpi=600, format='TIFF')
                        plt.clf()
        else:
            f = DVFile(input_dir + image_name)
            for image_num in range(1,3):
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
                    a = np.where(seg == i)   # somethin bad is happening when i = 4 on my tests
                    min_x = max(np.min(a[0]) - 1, 0)
                    max_x = min(np.max(a[0]) + 1, seg.shape[0])
                    min_y = max(np.min(a[1]) - 1, 0)
                    max_y = min(np.max(a[1]) + 1, seg.shape[1])
                    # a[0] contains the x coords and a[1] contains the y coords
                    # save this to use later when I want to calculate cellular intensity
                    #convert from absolute location to relative location for later use
                    if not os.path.exists(output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline')  or not use_cache:
                        with open(output_dir + 'masks/' + base_image_name + '-' + str(i) + '.outline', 'w') as csvfile:
                            csvwriter = csv.writer(csvfile, lineterminator='\n')
                            csvwriter.writerows(zip(a[0] - min_x, a[1] - min_y))
                    cellpair_image = image_outlined[min_x: max_x, min_y:max_y]
                    not_outlined_image = image[min_x: max_x, min_y:max_y]
                    if not os.path.exists(output_dir + 'segmented/' + cell_tif_image) or not use_cache:  # don't redo things we already have
                        plt.imsave(output_dir + 'segmented/' + cell_tif_image, cellpair_image, dpi=600, format='TIFF')
                        plt.clf()
                    if not os.path.exists(output_dir + 'segmented/' + no_outline_image) or not use_cache:  # don't redo things we already have
                        plt.imsave(output_dir + 'segmented/' + no_outline_image, not_outlined_image, dpi=600, format='TIFF')
                        plt.clf()
    # if the image_dict is empty, then we didn't get anything interesting from the directory
    print("image_dict123", image_dict)
    return image_dict
    # else: show error message
