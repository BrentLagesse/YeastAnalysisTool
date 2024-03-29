"""Module stats."""
from enum import Enum
import csv
import math
import cv2
import numpy as np
from PIL import Image
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import services
import main

global data
stat_plugins = list()

def add_stat_plugins(plugin):
    stat_plugins.append(plugin)


def load_image(image_path):
    return Image.open(image_path)

def preprocess_image(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def blur_image(image, kernel_size, kernel_deviation):
    kdev = int(kernel_deviation)
    ksize = int(kernel_size)
    if ksize % 2 == 0:
        ksize += 1
        print("You used an even ksize, updating to odd number +1")
    return cv2.GaussianBlur(image, (ksize, ksize), kdev)

def threshold_image(image):
    ret, thresh = cv2.threshold(image, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
    return thresh

def find_contours(image):
    contours, h = cv2.findContours(image, 1, 2)
    return contours

def intialize(cp, conf):
    global cp1, input_dir, output_dir, mcherry_line_width_input, kernel_deviation_input
    global contours, contours1, contours_mcherry, edit_testimg, contours_gfp, edit_GFP_img
    global edit_im, edit_im_GFP, orig_gray_GFP_no_bg, orig_gray_mcherry_no_bg
    cp1 = cp
    data = conf
    print(data)
    input_dir = data['input_dir']
    output_dir = data['output_dir']
    kernel_size_input = data['kernel_size']
    mcherry_line_width_input = data['mCherry_line_width']
    kernel_deviation_input = data['kernel_diviation']

    # outlines screw up the analysis
    im_cherry = load_image(f'{output_dir}/segmented/{cp1.get_mCherry(use_id=True, outline=False)}')
    im_GFP = load_image(f'{output_dir}/segmented/{cp1.get_GFP(use_id=True, outline=False)}')
    im_GFP_for_cellular_intensity = load_image(f'{output_dir}/segmented/{cp1.get_GFP(use_id=True)}')

    cell_intensity_gray = preprocess_image(im_GFP_for_cellular_intensity)

    # was RGBA2GRAY
    orig_gray_GFP = preprocess_image(im_GFP)
    orig_gray_GFP_no_bg, background = subtract_background_rolling_ball(orig_gray_GFP, 50, light_background=False,
                                                                       use_paraboloid=False, do_presmooth=True)

    orig_gray_mcherry = preprocess_image(im_cherry)
    orig_gray_mcherry_no_bg, backgroundmcherry = subtract_background_rolling_ball(orig_gray_mcherry, 50, light_background=False,
                                                                                  use_paraboloid=False, do_presmooth=True)

    orig_gray = preprocess_image(im_cherry)
    orig_GFP_gray = preprocess_image(im_GFP)

    gray_mcherry = cv2.GaussianBlur(orig_gray, (3, 3), 1)
    thresh_mcherry = threshold_image(gray_mcherry)
    global gray
    gray = blur_image(orig_gray, kernel_size_input, kernel_deviation_input)
    #print("gray test", gray)
    thresh = threshold_image(gray) 

    gray_gfp = cv2.GaussianBlur(orig_GFP_gray, (3, 3), 1)
    thresh_gfp = threshold_image(gray_gfp) 
    gray1 = blur_image(orig_GFP_gray, kernel_size_input, kernel_deviation_input)

    thresh1 = threshold_image(gray1)

    # Some of the cell outlines are split into two circles.  We blur this so that the contour will cover  both of them

    cell_intensity_gray = cv2.GaussianBlur(cell_intensity_gray, (3, 3), 1)

    cell_int_thresh = threshold_image(cell_intensity_gray) 
    cell_int_cont, cell_int_h = cv2.findContours(cell_int_thresh, 1, 2)

    # we want the biggest contour because that'll be our cellular intensity boundary
    largest = 0
    largest_cell_cnt = None
    # TODO:  I think I should join the two largest that aren't the full box
    for cnt in cell_int_cont:
        # if i == len(cell_int_cont) - 1:    # this is not robust
        # #TODO fix it  -- this throws out the full box contour
        #    continue
        area = cv2.contourArea(cnt)
        if area > largest:
            largest = area
            largest_cell_cnt = cnt

    contours, h = cv2.findContours(thresh, 1, 2)
    contours_mcherry = cv2.findContours(thresh_mcherry, 1, 2)

    contours1, h1 = cv2.findContours(thresh1, 1, 2)
    contours_gfp = cv2.findContours(thresh_gfp, 1, 2)
    # iterate through contours and throw out the largest (the box) and anything less than the second and third largest)
    # Contours finds the entire image as a contour and it seems to always put it in the contours[len(contours)].  We should do this more robustly in the future

    # these include the outlines already, so lets edit them
    edit_im = load_image(f'{output_dir}/segmented/{cp1.get_mCherry(use_id=True)}')
    edit_im_GFP = load_image(f'{output_dir}/segmented/{cp1.get_GFP(use_id=True)}')
    edit_testimg = np.array(edit_im)
    edit_GFP_img = np.array(edit_im_GFP)

def get_stats(cp, conf):
    #TODO:  Replace this code with a plugable processing system
    intialize(cp, conf)
    bestContours, mcherry_line_pts, best_contour, mcherry_distance, mcherry_count = calculate_bestContours(contours, contours_mcherry, edit_testimg, 'mCherry')
    bestContours1, gfp_line_pts, best_contour1, gfp_distance, gfp_count = calculate_bestContours(contours1, contours_gfp, edit_GFP_img, 'gfp')
    #print("test123123", bestContours1, bestContours)

    intensity_sum = find_intensity_sum(bestContours, best_contour)
    cell_intensity_sum = find_cell_intensity_sum()
    mcherry_edit_image, mcherry_line_intensity_sum = find_mcherry_line_GFP_intensity(bestContours, mcherry_line_pts, best_contour)
    gfp_edit_im_image, GFP_line_intensity_sum = find_GFP_line_GFP_intensity(bestContours1, gfp_line_pts, best_contour1)

    cp.red_dot_distance = mcherry_distance
    cp.red_dot_count = mcherry_count
    cp.gfp_distance = gfp_distance
    cp.gfp_line_gfp_intensity = GFP_line_intensity_sum
    cp.mcherry_line_gfp_intensity = mcherry_line_intensity_sum
    #cp.set_GFP_Cell_Intensity(main.Contour.CONTOUR, intensity_sum, intensity_pts)
    cp.nucleus_intensity[main.Contour.CONTOUR] = intensity_sum
    cp.cell_intensity = cell_intensity_sum


    cp.set_property('gfp_distance', gfp_distance)
    cp.set_property('gfp_intensity', GFP_line_intensity_sum)

    cp.set_property('nucleus_intensity', intensity_sum)
    cp.set_property('cell_intensity', cell_intensity_sum)
    cytoplasmic_intensity = cell_intensity_sum - intensity_sum
    cp.set_property('cytoplasmic_intensity', cytoplasmic_intensity)
    cp.set_property('nuc_cyto_ratio', float(intensity_sum) / float(cytoplasmic_intensity))


#TODO: This is going to replace the hard coded stats above
    for stat in stat_plugins:
        if not stat.ENABLED:
            continue
        #TODO: figure out if we have the data
        # NOTE:  hardcoded for now
        data_for_plugin = dict()
        # Remove later
        data_for_plugin['contours'] = contours
        data_for_plugin['contours_mcherry'] = contours_mcherry
        data_for_plugin['countours1'] = contours1
        data_for_plugin['contours_gfp'] = contours_gfp
        data_for_plugin['edit_testimg'] = edit_testimg
        data_for_plugin['orig_gray_mcherry_no_bg'] = orig_gray_mcherry_no_bg
        for request in stat.required_data():
            if data_for_plugin.get(request) is None:
                pass  # TODO:  Identify the analysis that can provide this and run it
        # at this point we know we have all the required data processed
        # run the stats
        new_stats = stat.return_stats(data_for_plugin)
        # add the stats to our cellpair
        cp.set_properties(new_stats)


    return mcherry_edit_image, gfp_edit_im_image, intensity_sum, cell_intensity_sum, mcherry_distance, mcherry_count, gfp_distance, gfp_count, mcherry_line_intensity_sum, GFP_line_intensity_sum


def find_mcherry_line_GFP_intensity(bestContours, mcherry_line_pts, best_contour):
    if bestContours == 0:
        return edit_im, 0
    elif len(bestContours) == 1:	
        best_contour = contours[bestContours[0]]
    #print("only 1 contour found")
    cv2.drawContours(edit_testimg, [best_contour], 0, (0, 255, 0), 1)
    mcherry_line_intensity_sum = sum(
        orig_gray_mcherry_no_bg[p[0]][p[1]] for p in mcherry_line_pts
    )
    # cp1.set_mcherry_line_GFP_intensity(mcherry_line_intensity_sum)


    return Image.fromarray(edit_testimg), mcherry_line_intensity_sum

def find_GFP_line_GFP_intensity(bestContours1, gfp_line_pts, best_contour1):
    if bestContours1 == 0:
        return edit_im_GFP, 0

    if len(bestContours1) == 1:	
        best_contour1 = contours[bestContours1[0]]
    #print("only 1 contour found")
    cv2.drawContours(edit_GFP_img, [best_contour1], 0, (0, 255, 0), 1)	

    GFP_line_intensity_sum = sum(
        orig_gray_GFP_no_bg[p[0]][p[1]] for p in gfp_line_pts
    )
   

    return Image.fromarray(edit_GFP_img), GFP_line_intensity_sum

def find_intensity_sum(bestContours, best_contour):
    if bestContours == 0:
        #return edit_im   # TODO:  Why is this returning an image?
        return 0
    elif len(bestContours) == 1:	
        best_contour = contours[bestContours[0]]
    
    mask_contour = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask_contour, [best_contour], 255)
    #print("mask_contour", mask_contour)
    pts_contour = np.transpose(np.nonzero(mask_contour))
    #print("pts_contour", pts_contour)
    intensity_sum = sum(orig_gray_GFP_no_bg[p[0]][p[1]] for p in pts_contour)
    # cp1.set_GFP_Nucleus_Intensity(
    #     main.Contour.CONTOUR, intensity_sum, len(pts_contour))
    return intensity_sum
    
    
def find_cell_intensity_sum():
    # read in the outline file if you need it
    border_cells = []
    with open(f'{output_dir}/masks/{cp1.get_base_name()}-{str(cp1.id)}.outline', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        border_cells.extend([int(row[0]), int(row[1])] for row in csvreader)
    return sum(orig_gray_GFP_no_bg[p[0]][p[1]] for p in border_cells)

def get_best_contours(contours):
    bestContours = []
    bestArea = []
    for i, cnt in enumerate(contours):
        if len(cnt) == 0:
            continue
        if i == len(contours) - 1:    # this is not robust #TODO fix it
            continue
        try:
            area = cv2.contourArea(cnt)
        except Exception:
            continue

        if not bestContours:
            bestContours.append(i)
            bestArea.append(area)
            continue
        if len(bestContours) == 1:
            bestContours.append(i)
            bestArea.append(area)
        if area > bestArea[0]:
            best_area_contours(bestArea, area, bestContours, i)
        elif area > bestArea[1]:
            bestArea[1] = area
            bestContours[1] = i
    return bestContours, bestArea

def calculate_bestContours(contours, contours_mcherry, edit_testimg, type):
    best_contour = []
    bestContours, bestArea = get_best_contours(contours)
    distance = 0
    count = 0
    if not bestContours:
        #print("we didn't find any contours")
        return 0,0,0,0,0

    bestContours_mcherry, bestArea_mcherry = get_best_contours(contours_mcherry[0])

    mcherry_line_pts = []
    if len(bestContours_mcherry) == 2:
        c1 = contours_mcherry[0][bestContours_mcherry[0]]
        c2 = contours_mcherry[0][bestContours_mcherry[1]]
        M1 = cv2.moments(c1)
        M2 = cv2.moments(c2)
        # TODO:  This was code from when we wanted to get the distance between the mCherry centers
        if M1['m00'] == 0 or M2['m00'] == 0:   # something has gone wrong
            print("Warning:  The m00 moment = 0")
            # plt.imshow(edit_testimg,  cmap='gray')
            # plt.show()
        else:
            mcherry_line_pts, distance, count = get_mcherry_line_pts(
                M1, M2, type, edit_testimg
            )
    if len(bestContours) == 2:  # "There can be only one!" - Connor MacLeod
        c1 = contours[bestContours[0]]
        c2 = contours[bestContours[1]]
        MERGE_CLOSEST = True
        if MERGE_CLOSEST:   # find the two closest points and just push c2 into c1 there
            best_contour = find_best_contours(c1, c2)
    return bestContours, mcherry_line_pts, best_contour, distance, count

def find_best_contours(c1, c2):
    smallest_distance = 999999999
    second_smallest_distance = 999999999
    # invalid so it'll cause an error if it is used
    smallest_pair = (-1, -1)
    for pt1 in c1:
        for i, pt2 in enumerate(c2):
            d = math.sqrt(
                pow(pt1[0][0] - pt2[0][0], 2) + pow(pt1[0][1] - pt2[0][1], 2))
            if d < smallest_distance:
                second_smallest_distance = smallest_distance
                second_smallest_pair = smallest_pair
                smallest_distance = d
                smallest_pair = (pt1, pt2, i)
            elif d < second_smallest_distance:
                second_smallest_distance = d
                second_smallest_pair = (pt1, pt2, i)

    # now we have the two closest points to each other between the two contours
    # iterate through the contour 1 until you find pt1, then add every point in c2
    # to c1, starting from pt2 until you reach pt2 of the second_smallest pair, then
    # remove points in c1 until you reach pt1 of second_smallest_pair
    clockwise = True  # we need to figure out which  of the two points shoudl go first

    global result
    result = []
            # temp hacky way of doing it
    for pt1 in c1:
        result.append(pt1)
        if pt1[0].tolist() != smallest_pair[0][0].tolist():
            continue
        # we are at the closest p1
        start_loc = smallest_pair[2]
        finish_loc = start_loc - 1
        if start_loc == 0:
            finish_loc = len(c2) - 1
        current_loc = start_loc
        while current_loc != finish_loc:
            result.append(c2[current_loc])
            current_loc += 1
            if current_loc >= len(c2):
                current_loc = 0
                # grab the last point
        result.append(c2[finish_loc])
    return np.array(result).reshape((-1, 1, 2)).astype(np.int32)

def get_mcherry_line_pts(M1, M2, type, edit_testimg):
    global mcherry_distance, mcherry_count, gfp_distance, gfp_count
    mcherry_distance, mcherry_count, gfp_distance, gfp_count = 0, 0, 0, 0
 #   print("type", type)
    c1x, c1y = services.getMoments(M1)
    c2x, c2y = services.getMoments(M2)
    d = math.sqrt(pow(c1x - c2x, 2) + pow(c1y - c2y, 2))
    # print ('Distance: ' + str(d))
    if type == 'mCherry':
        mcherry_distance = d
        mcherry_count = 2
    else:
        gfp_distance = d
        gfp_count = 2
    draw_circle_line(edit_testimg, c1x, c1y, c2x, c2y)
    mcherry_line_mask = np.zeros(gray.shape, np.uint8)
    draw_circle_line(mcherry_line_mask, c1x, c1y, c2x, c2y)
    return np.transpose(np.nonzero(mcherry_line_mask)), d, 2

def best_area_contours(arg0, area, arg2, i):
    arg0[1] = arg0[0]
    arg0[0] = area
    arg2[1] = arg2[0]
    arg2[0] = i

def draw_circle_line(img, c1x, c1y, c2x, c2y):
    color = 255
    radius = 1
    thickness = 2
    cv2.circle(img, (c1x, c1y), radius, color, thickness)
    cv2.circle(img, (c2x, c2y), radius, color, thickness)
    cv2.line(img, (c1x, c1y), (c2x, c2y),
                color, int(mcherry_line_width_input))