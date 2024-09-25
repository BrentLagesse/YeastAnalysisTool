import StatPlugin
import stats
import cv2
from PIL import Image

class MCherry(StatPlugin.StatPlugin):
    ENABLED = True

    def required_data(self):
        return list()

    def return_stats(self, data):
        result = dict()
        contours = data['contours']
        contours_mcherry = data['contours_mcherry']
        edit_testimg= data['edit_testimg']
        orig_gray_mcherry_no_bg = data['orig_gray_mcherry_no_bg']
        bestContours, mcherry_line_pts, best_contour, mcherry_distance, mcherry_count = stats.calculate_bestContours(contours, contours_mcherry, edit_testimg, 'mCherry')

        if bestContours == 0:
            result['mcherry_line_intensity_sum'] = None
            result['mcherry_distance'] = None
            result['mcherry_count'] = None
            return result
        elif len(bestContours) == 1:
            best_contour = contours[bestContours[0]]
        # print("only 1 contour found")
        cv2.drawContours(edit_testimg, [best_contour], 0, (0, 255, 0), 1)
        mcherry_line_intensity_sum = sum(
            orig_gray_mcherry_no_bg[p[0]][p[1]] for p in mcherry_line_pts
        )
        # cp1.set_mcherry_line_GFP_intensity(mcherry_line_intensity_sum)

        result['mcherry_line_intensity_sum'] = mcherry_line_intensity_sum
        result['mcherry_distance'] = mcherry_distance
        result['mcherry_count'] = mcherry_count
        return result

    def where_to_diplay(self):
        return None