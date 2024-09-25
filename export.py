import pytz
from tkinter import *
import customtkinter
from tkinter import filedialog
from tkinter.ttk import *

import csv
import main

import stats


outline_dict = dict()  # store the outlines so I don't have to reprocess them

image_dict = dict()    # image -> list of cell ids
cp_dict = dict()   

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
        outfile_writer.writerow(['imagename', 'cellid',  'datetime', 'kernel size', 'kernel deviation', 'mcherry line width', 'contour type', 'thesholding options', 'nuclear GFP', 'cellular GFP', 'cytoplasmic intensity', 'nuc int/cyto int', 'mcherry distance', 'mcherry line gfp intensity', 'user invalidated'])
        for image, cells  in image_dict.items():
            for cell in cells:
                cp = cp_dict.get((image, cell))
                if cp == None:   # create a new one and run stats
                    cp = main.CellPair(image, cell) 
                    stats.get_stats(cp,data)
                    cp_dict[(image, cell)] = cp

                if cp.get_ignored() and drop_ignored.get():
                    continue
                line = list()  #all the elements to write
                try:
                    nucleus_intensity = cp.get_GFP_Nucleus_Intensity(main.Contour.CONTOUR)[0]
                    cellular_intensity = cp.get_GFP_Cell_Intensity()[0]
                    cytoplasmic_intensity = cellular_intensity - nucleus_intensity
                    nuc_div_cyto_intensity = float(nucleus_intensity)/float(cytoplasmic_intensity)
                except:
                    print('Invalid values in image ' + str(image) + '  and cell ' + str(cell) + '... skipping cell')
                    continue
                line.append(image)
                line.append(cell)
                line.append(now)
                line.append(kernel_size_input)
                line.append(kernel_deviation_input)
                line.append(mcherry_line_width_input)
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