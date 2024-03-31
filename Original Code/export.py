import pytz
from tkinter import *
import customtkinter
from tkinter import filedialog
from tkinter.ttk import *

import csv
import main

import stats


outline_dict = {}

image_dict = {}
cp_dict = {}   

def export_to_csv_file(conf,window, image_dict1,cp_dict1, drop_ignored1):
    image_dict = image_dict1
    cp_dict = cp_dict1
    drop_ignored = drop_ignored1
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


    from datetime import datetime
    now = datetime.utcnow().replace(tzinfo=pytz.utc)

    csv_out = filedialog.asksaveasfilename(parent=window, title='Save as...', initialdir='.', defaultextension='.csv')
    with open(csv_out, mode='w') as outfile:
        outfile_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #write headers
        # Image name, cell id, date, time, software version number, thresholding technique, contour technique, smoothing technique
        headers = ['imagename', 'cellid', 'datetime', 'kernel size', 'kernel deviation', 'mcherry line width', 'contour type',
         'thesholding options',]

        first_row = True
       # outfile_writer.writerow(['imagename', 'cellid',  'datetime', 'kernel size', 'kernel deviation', 'mcherry line width', 'contour type', 'thesholding options', 'nuclear GFP', 'cellular GFP', 'cytoplasmic intensity', 'nuc int/cyto int', 'mcherry distance', 'mcherry line gfp intensity', 'user invalidated'])
        for image, cells in image_dict.items():
            for cell in cells:
                cp = cp_dict.get((image, cell))

                if cp is None:   # create a new one and run stats
                    cp = main.CellPair(image, cell) 
                    stats.get_stats(cp,data)
                    cp_dict[(image, cell)] = cp
                property_keys = list(cp.get_all_properties().keys())   # TODO:  we should do this outside the loop and not depend on this always returning values in the same order
                if first_row:
                    headers += property_keys
                    headers += ['user invalidated']
                    outfile_writer.writerow(headers)
                    first_row = False
                if cp.get_ignored() and drop_ignored.get():
                    continue

                line = [
                    image,
                    cell,
                    now,
                    kernel_size_input,
                    kernel_deviation_input,
                    mcherry_line_width_input,
                    'contour',
                    'cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU',]
                for k in property_keys:
                    line += [cp.get_all_properties().get(k)]

                line += [cp.get_ignored()]

                outfile_writer.writerow(line)

