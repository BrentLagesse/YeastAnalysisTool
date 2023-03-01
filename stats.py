import cv2
import numpy as np
from PIL import ImageTk,Image
import csv
import math
from enum import Enum
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt

import main
global data


def get_stats(cp, conf):
    data = conf
    print(data)
    global input_dir 
    input_dir= data['input_dir']
    global output_dir 
    output_dir = data['output_dir']
    kernel_size_input = data['kernel_size']
    mcherry_line_width_input = data['mCherry_line_width']
    kernel_deviation_input = data['kernel_diviation']
    choice_var = data['arrested']
    
    #outlines screw up the analysis
    print("test123", 'segmented/' + cp.get_mCherry(use_id=True, outline=False))
    im = Image.open(output_dir + '/segmented/' + cp.get_mCherry(use_id=True, outline=False))
    im_cherry = Image.open(output_dir + '/segmented/' + cp.get_mCherry(use_id=True, outline=False))
    im_GFP = Image.open(output_dir + '/segmented/' + cp.get_GFP(use_id=True, outline=False))
    im_GFP_for_cellular_intensity = Image.open(output_dir + '/segmented/' + cp.get_GFP(use_id=True))  #has outline
    testimg = np.array(im)
    GFP_img = np.array(im_GFP)
    mcherry_img = np.array(im_cherry)
    img_for_cell_intensity = np.array(im_GFP_for_cellular_intensity)

    cell_intensity_gray = cv2.cvtColor(img_for_cell_intensity, cv2.COLOR_RGB2GRAY)

    # was RGBA2GRAY
    orig_gray_GFP = cv2.cvtColor(GFP_img, cv2.COLOR_RGB2GRAY)
    orig_gray_GFP_no_bg, background = subtract_background_rolling_ball(orig_gray_GFP, 50, light_background=False,
                                                       use_paraboloid=False, do_presmooth=True)

    orig_gray_mcherry = cv2.cvtColor(mcherry_img, cv2.COLOR_RGB2GRAY)
    orig_gray_mcherry_no_bg, backgroundmcherry = subtract_background_rolling_ball(orig_gray_mcherry, 50, light_background=False,
                                                                       use_paraboloid=False, do_presmooth=True)
    orig_gray = cv2.cvtColor(testimg, cv2.COLOR_RGB2GRAY)
    orig_GFP_gray = cv2.cvtColor(GFP_img, cv2.COLOR_RGB2GRAY)
    kdev = int(kernel_deviation_input)
    ksize = int(kernel_size_input)
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

    gray_gfp = cv2.GaussianBlur(orig_GFP_gray, (3, 3), 1)
    ret_gfp, thresh_gfp = cv2.threshold(gray_gfp, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
    gray1 = cv2.GaussianBlur(orig_GFP_gray, (ksize, ksize), kdev)
    # plt.title("blur")
    # plt.imshow(gray,  cmap='gray')
    # plt.show()
    ret1, thresh1 = cv2.threshold(gray1, 0, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)


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

    contours1, h1 = cv2.findContours(thresh1, 1, 2)
    contours_gfp = cv2.findContours(thresh_gfp, 1, 2)
    #iterate through contours and throw out the largest (the box) and anything less than the second and third largest)
    # Contours finds the entire image as a contour and it seems to always put it in the contours[len(contours)].  We should do this more robustly in the future



    #bestContours, bestArea = find_best_contours(contours)
    #bestContours_mcherry, bestArea_mcherry = find_best_contours(contours_mcherry)

    #these include the outlines already, so lets edit them
    edit_im = Image.open(output_dir + '/segmented/' + cp.get_mCherry(use_id=True))
    edit_im_GFP = Image.open(output_dir + '/segmented/' + cp.get_GFP(use_id=True))
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

    best_contour1 = None

    bestContours1 = list()
    bestArea1 = list()
    for i, cnt in enumerate(contours1):
        #tester = orig_gray
        if len(cnt) == 0:
            continue
        #cv2.drawContours(tester, cnt, 0, 255, 1)
        #plt.imshow(tester,  cmap='gray')
        #plt.show()
        if i == len(contours1) - 1:    # this is not robust #TODO fix it
            continue
        area1 = cv2.contourArea(cnt)

        if len(bestContours1) == 0:
            bestContours1.append(i)
            bestArea1.append(area)
            continue
        if len(bestContours1) == 1:
            bestContours1.append(i)
            bestArea1.append(area)
        if area1 > bestArea1[0]:
            bestArea1[1] = bestArea1[0]
            bestArea1[0] = area1
            bestContours1[1] = bestContours1[0]
            bestContours1[0] = i
        elif area1 > bestArea[1]:    # probably won't have a 3rd that is equal, but that would cause a problem
            bestArea[1] = area1
            bestContours[1] = i


    if len(bestContours1) == 0:
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
            cv2.line(edit_testimg, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input))
            mcherry_line_mask = np.zeros(gray.shape, np.uint8)
            cv2.line(mcherry_line_mask, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input))
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

    bestContours_gfp = list()
    bestArea_gfp = list()
    for i, cnt in enumerate(contours_gfp[0]):
        # tester = orig_gray
        if len(cnt) == 0:
            continue
        # cv2.drawContours(tester, cnt, 0, 255, 1)
        # plt.imshow(tester,  cmap='gray')
        # plt.show()
        if i == len(contours_gfp[0]) - 1:  # this is not robust #TODO fix it
            continue
        try:
            area1 = cv2.contourArea(cnt)
        except:  # no area
            continue

        if len(bestContours_gfp) == 0:
            bestContours_gfp.append(i)
            bestArea_gfp.append(area)
            continue
        if len(bestContours_gfp) == 1:
            bestContours_gfp.append(i)
            bestArea_gfp.append(area)
        if area1 > bestArea_gfp[0]:
            bestArea_gfp[1] = bestArea_gfp[0]
            bestArea_gfp[0] = area1
            bestContours_gfp[1] = bestContours_gfp[0]
            bestContours_gfp[0] = i
        elif area1 > bestArea_gfp[1]:  # probably won't have a 3rd that is equal, but that would cause a problem
            bestArea_gfp[1] = area1
            bestContours_gfp[1] = i

    gfp_line_pts = list()
    if len(bestContours_gfp) == 2:
        c1 = contours_gfp[0][bestContours_gfp[0]]
        c2 = contours_gfp[0][bestContours_gfp[1]]
        M1 = cv2.moments(c1)
        M2 = cv2.moments(c2)
        # TODO:  This was code from when we wanted to get the distance between the mCherry centers
        if M1['m00'] == 0 or M2['m00'] == 0:  # something has gone wrong
            print("Warning:  The m00 moment = 0")
            # plt.imshow(edit_testimg,  cmap='gray')
            # plt.show()
        else:

            c1x = int(M1['m10'] / M1['m00'])
            c1y = int(M1['m01'] / M1['m00'])
            c2x = int(M2['m10'] / M2['m00'])
            c2y = int(M2['m01'] / M2['m00'])
            d = math.sqrt(pow(c1x - c2x, 2) + pow(c1y - c2y, 2))
            print('GFP Distance: ' + str(d))
            cp.set_gfp_red_dot_distance(d)
            print("cordinates", (c1x, c1y), (c2x, c2y))
            cv2.line(edit_GFP_img, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input))
            gfp_line_mask = np.zeros(gray_gfp.shape, np.uint8)
            cv2.line(gfp_line_mask, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input))
            gfp_line_pts = np.transpose(np.nonzero(gfp_line_mask))

    if len(bestContours1) == 2:  # "There can be only one!" - Connor MacLeod
        c1 = contours[bestContours1[0]]
        c2 = contours[bestContours1[1]]
        MERGE_CLOSEST = True
        if MERGE_CLOSEST:  # find the two closest points and just push c2 into c1 there
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

            # now we have the two closest points to each other between the two contours
            # iterate through the contour 1 until you find pt1, then add every point in c2
            # to c1, starting from pt2 until you reach pt2 of the second_smallest pair, then
            # remove points in c1 until you reach pt1 of second_smallest_pair
            clockwise = True  # we need to figure out which  of the two points shoudl go first

            best_contour1 = list()
            # temp hacky way of doing it
            for pt1 in c1:
                best_contour1.append(pt1)
                if pt1[0].tolist() != smallest_pair[0][0].tolist():
                    continue
                # we are at the closest p1
                start_loc = smallest_pair[2]
                finish_loc = start_loc - 1
                if start_loc == 0:
                    finish_loc = len(c2) - 1
                current_loc = start_loc
                while current_loc != finish_loc:
                    best_contour1.append(c2[current_loc])
                    current_loc += 1
                    if current_loc >= len(c2):
                        current_loc = 0
                # grab the last point
                best_contour1.append(c2[finish_loc])
            best_contour1 = np.array(best_contour1).reshape((-1, 1, 2)).astype(np.int32)







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
    with open(output_dir + '/masks/' + cp.get_base_name() + '-' + str(cp.id) + '.outline', 'r') as csvfile:
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
    cp.set_GFP_Nucleus_Intensity(main.Contour.CONTOUR, intensity_sum, len(pts_contour))



    cell_intensity_sum = 0
    for p in border_cells:
        cell_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]
    cp.set_GFP_Cell_Intensity(cell_intensity_sum, len(border_cells))


    mcherry_line_intensity_sum = 0

    for p in mcherry_line_pts:
        mcherry_line_intensity_sum += orig_gray_mcherry_no_bg[p[0]][p[1]]
    cp.set_mcherry_line_GFP_intensity(mcherry_line_intensity_sum)

    GFP_line_intensity_sum = 0

    for p in gfp_line_pts:
        GFP_line_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]
    cp.set_GFP_line_GFP_intensity(GFP_line_intensity_sum)






    return Image.fromarray(edit_testimg), Image.fromarray(edit_GFP_img)
