from tkinter import *
import customtkinter
from tkinter.ttk import *
from functools import partial
from scipy.spatial import distance as dist

import opts as opt
import os

from PIL import ImageTk, Image

from mrc import DVFile
from enum import Enum
import stats
import segment_images
from export import export_to_csv_file

input_dir = opt.input_directory
output_dir = opt.output_directory
ignore_btn = None

current_image = None
current_cell = None

outline_dict = {}

global image_dict
image_dict = {}
cp_dict = {}

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
        self.gfp_dot_count = 0
        self.red_dot_distance = 0
        self.gfp_red_dot_distance = 0
        self.cyan_dot_count = 1
        self.green_dot_count = 1
        self.ground_truth = False
        self.nucleus_intensity = {}
        self.nucleus_total_points = 0
        self.cell_intensity = {}
        self.cell_total_points = 0
        self.ignored = False
        self.mcherry_line_gfp_intensity = 0
        self.gfp_line_gfp_intensity = 0

    def set_red_dot_distance(self, d):
        self.red_dot_distance = d

    def set_gfp_red_dot_distance(self, d):
        self.gfp_red_dot_distance = d

    def set_red_dot_count(self, c):
        self.red_dot_count = c
    
    def get_red_dot_count(self):
        return self.red_dot_count

    def set_gfp_dot_count(self, c):
        self.gfp_dot_count = c
    
    def get_gfp_dot_count(self):
        return self.gfp_dot_count

    def get_base_name(self):
        print("imagetest:", self.image_name)
        if img_format == "tiff":
            return self.image_name.split('_R3D_REF')[0]
        else:
            return self.image_name.split('_PRJ')[0]

    def get_DIC(self, use_id=False, outline=True, segmented=False, main_img=False):
        if img_format == 'tiff':
            if not use_id:
                return f'{self.get_base_name()}_R3D_REF.tif'
            outlinestr = '' if outline else '-no_outline'
            return f'{self.get_base_name()}_R3D_REF-{str(self.id)}{outlinestr}.tif'
        else:
            outlinestr = '' if outline else '-no_outline'
            if main_img:
                return Image.open(f'{output_dir}/segmented/{self.get_base_name()}_PRJ.tif')
            if segmented:
                return Image.open(
                    f'{output_dir}/segmented/{self.get_base_name()}_PRJ-{str(self.id)}{outlinestr}.tif'
                )
            if use_id:
                image_loc = f'{output_dir}/segmented/{self.get_base_name()}_PRJ-{str(self.id)}{outlinestr}.tif'
                return Image.open(image_loc)
            else:
                # look for dv file,
                # open dv file if exists,
                # return the appropriate image from the stack (actual image)
                extspl = os.path.splitext(self.image_name)
                if extspl[1] == '.dv':
                    f = DVFile(input_dir + self.image_name)
                    image = f.asarray()
                    return Image.fromarray(image[0])
                else:
                    image_loc = f'{output_dir}/segmented/{self.get_base_name()}_PRJ{outlinestr}.tif'
                    return Image.open(image_loc)

    def get_DAPI(self, use_id=False, outline=True):
        outlinestr = '' if outline else '-no_outline'
        if img_format == 'tiff':
            return (
                f'{self.get_base_name()}_PRJ_w435-{str(self.id)}{outlinestr}.tif'
                if use_id
                else f'{self.get_base_name()}_PRJ_w435{outlinestr}.tif'
            )
            # check if there are .dv files and use them first
        if not use_id:
            image_loc = f'{output_dir}/segmented/{self.get_base_name()}_PRJ-{str(self.id)}{outlinestr}.tif'
            return Image.open(image_loc)
        else:
            extspl = os.path.splitext(self.image_name)
            if extspl[1] == '.dv':
                f = DVFile(input_dir + self.image_name)
                image = f.asarray()
                return Image.fromarray(image[1])
            else:
                image_loc = f'{output_dir}/segmented/{self.get_base_name()}_PRJ{outlinestr}.tif'
                return Image.open(image_loc)

    def get_GFP(self, use_id=False, outline=True):
        outlinestr = '' if outline else '-no_outline'
        if img_format == 'tiff':
            return (
                f'{self.get_base_name()}_PRJ_w525-{str(self.id)}{outlinestr}.tif'
                if use_id
                else f'{self.get_base_name()}_PRJ_w525{outlinestr}.tif'
            )
        if use_id:
            return f'{self.get_base_name()}_PRJ-{str(self.id)}{outlinestr}.tif'

        extspl = os.path.splitext(self.image_name)
        if extspl[1] != '.dv':
            return f'{self.get_base_name()}_PRJ{outlinestr}.tif'
        f = DVFile(input_dir + self.image_name)
        image = f.asarray()
        return Image.fromarray(image[2])

    def set_GFP_Nucleus_Intensity(self, contour_type, val, total_points):
        self.nucleus_intensity[contour_type] = val
        self.nucleus_total_points = total_points

    def set_GFP_Cell_Intensity(self, val, total_points):
        self.cell_intensity = val
        self.cell_total_points = total_points

    def get_GFP_Nucleus_Intensity(self, contour_type):
        if self.nucleus_intensity.get(contour_type) is None:
            return (0, 0)
        if self.nucleus_intensity[contour_type] == 0:  # this causes an error if nothing has been set.  This is expected
            print("Intensity is 0, this is unlikely")
        return self.nucleus_intensity[contour_type], self.nucleus_total_points

    def get_GFP_Cell_Intensity(self):
        if self.cell_intensity == 0:
            print("Intensity is 0, this is unlikely")
        return self.cell_intensity, self.cell_total_points

    def set_mcherry_line_GFP_intensity(self, intensity):
        self.mcherry_line_gfp_intensity = intensity

    def set_GFP_line_GFP_intensity(self, intensity):
        self.gfp_line_gfp_intensity = intensity

    def get_GFP_line_GFP_intensity(self):
        return self.gfp_line_gfp_intensity

    def get_mcherry_line_GFP_intensity(self):
        return self.mcherry_line_gfp_intensity

    def get_mCherry(self, use_id=False, outline=True):
        outlinestr = '' if outline else '-no_outline'
        if img_format == 'tiff':
            return (
                f'{self.get_base_name()}_PRJ_w625-{str(self.id)}{outlinestr}.tif'
                if use_id
                else f'{self.get_base_name()}_PRJ_w625{outlinestr}.tif'
            )
        if use_id:
            return f'{self.get_base_name()}_PRJ-{str(self.id)}{outlinestr}.tif'
        extspl = os.path.splitext(self.image_name)
        if extspl[1] != '.dv':
                    #return output_dir + 'segmented/' + self.get_base_name() + '_PRJ' + '_w625' + outlinestr + '.tif'
                    #return Image.open(image_loc)
            return f'{self.get_base_name()}_PRJ{outlinestr}.tif'
        f = DVFile(input_dir + self.image_name)
        image = f.asarray()
        return Image.fromarray(image[3])

    # TODO: Remove is Matt says we will never get this one
    def get_CFP(self, use_id=False, outline=True):
        outlinestr = '' if outline else '-no_outline'
        if img_format == 'tiff':
            return (
                f'{self.get_base_name()}/{self.get_base_name()}_PRJ_TIFFS/{self.get_base_name()}_w435-{str(self.id)}{outlinestr}.tif'
                if use_id
                else f'{self.get_base_name()}/{self.get_base_name()}_PRJ_TIFFS/{self.get_base_name()}_w435{outlinestr}.tif'
            )
        if use_id:
            return f'{self.get_base_name()}/{self.get_base_name()}_PRJ_TIFFS/{self.get_base_name()}-{str(self.id)}{outlinestr}.tif'
        #return self.get_base_name() + '/' + self.get_base_name() + '_PRJ_TIFFS/' + self.get_base_name() + '_w435' + outlinestr + '.tif'
        # look for dv file,
        # open dv file if exists,
        # return the appropriate image from the stack (actual image)
        extspl = os.path.splitext(self.image_name)
        if extspl[1] != '.dv':
            return f'{self.get_base_name()}/{self.get_base_name()}_PRJ_TIFFS/{self.get_base_name()}{outlinestr}.tif'
        f = DVFile(input_dir + self.image_name)
        image = f.asarray()
        return Image.fromarray(image[1])

    def get_member_variables(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def get_values_as_csv(self):
        # get all the member variables
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


def export_to_csv():
    global image_dict
    global cp_dict
    global drop_ignored
    export_to_csv_file(data, window, image_dict, cp_dict, drop_ignored)


def ignore(image, id):
    global ignore_btn
    global cp_dict
    cp_dict[(image, id)].set_ignored(not cp_dict[(image, id)].get_ignored())
    if cp_dict[(image, id)].get_ignored():
        ignore_btn.configure(text='ENABLE')
    else:
        ignore_btn.configure(text='IGNORE')

    # attempt to get distance
    # testimg = cv2.imread(image_loc, cv2.IMREAD_UNCHANGED)


# TODO:  Deal with resizing
def on_resize(event):
    try:
        w = width
        h = height
    except Exception:
        return


def display_cell(image, id):
    global ignore_btn, current_image, current_cell, export_btn, drop_ignored_checkbox, window, data
    export_btn['state'] = NORMAL
    drop_ignored_checkbox['state'] = NORMAL

    current_image = image
    current_cell = id

    win_width = window.winfo_width()
    win_height = window.winfo_height()
    print("image123", image)
    print("image_dict123", image_dict)
    max_id = len(image_dict[image])
    if max_id == 0:
        print('No cells found in this image')
        return
    if id < 1:
        id = max_id
    elif id > max_id:
        id = 1
    ID_label.configure(text=f'Cell ID:  {str(id)}')
    img_title_label.configure(text=image)
    print("displayImagename", image, cp_dict)
    cp = cp_dict.get((image, id))
    print("cp123", cp)
    if cp is None:
        cp = CellPair(image, id)
        cp_dict[(image, id)] = cp

    main_size_x = int(0.5 * win_width)
    main_size_y = int(0.5 * win_height)
    cell_size_x = int(0.23 * main_size_x)
    cell_size_y = int(0.23 * main_size_x)

    # DIC Image
    im_cherry, im_gfp = stats.get_stats(cp, data)
    if img_format == 'tiff':
        im = Image.open(f'{output_dir}/segmented/{cp.get_DIC()}')
    else:
        im = cp.get_DIC(main_img=True)
    img = resize_image(im, main_size_x, main_size_y, img_format)
    img_label.configure(image=img)
    img_label.image = img

    # DIC Label
    if img_format == 'tiff':
        image_loc = f'{output_dir}/segmented/{cp.get_DIC(use_id=True)}'
        im = Image.open(image_loc)
    else:
        im = cp.get_DIC(segmented=True)
    img = resize_image(im, cell_size_x, cell_size_y, img_format)
    DIC_label.configure(image=img)
    DIC_label.image = img
    DIC_label_text.configure(text="DIC")

    # DAPI Label
    if img_format == 'tiff':
        image_loc = f'{output_dir}/segmented/{cp.get_DAPI(use_id=True)}'
        im = Image.open(image_loc)
    else:
        im = cp.get_DAPI()
    img = resize_image(im, cell_size_x, cell_size_y, img_format)
    DAPI_label.configure(image=img)
    DAPI_label.image = img
    DAPI_label_text.configure(text="DAPI")

    # mCherry Label
    im = im_cherry
    img = resize_image(im, cell_size_x, cell_size_y, img_format)
    mCherry_label.configure(image=img)
    mCherry_label.image = img
    mCherry_label_text.configure(text="mCherry")

    # GFP Label
    im = im_gfp
    img = resize_image(im, cell_size_x, cell_size_y, img_format)
    GFP_label.configure(image=img)
    GFP_label.image = img
    GFP_label_text.configure(text="GFP")

    nuclei_count = 0
    dist_mcherry = customtkinter.CTkLabel(window)
    dist_mcherry.configure(text="Distance: {:.3f}".format(cp.red_dot_distance))
    # rad3.grid(row=6, column=3)
    # rad4.grid(row=7, column=3)
    dist_mcherry.grid(row=7, column=3)

    intensity_mcherry_lbl = customtkinter.CTkLabel(window)
    intensity_mcherry_lbl.configure(
        text=f"Line GFP intensity: {cp.get_mcherry_line_GFP_intensity()}"
    )
    intensity_mcherry_lbl.grid(row=8, column=3)

    red_dot_count_diplay = customtkinter.CTkLabel(window)
    red_dot_count_diplay.configure(text=f"red dot count:{cp.get_red_dot_count()}")
    red_dot_count_diplay.grid(row=9, column=3)

    try:
        intense1 = customtkinter.CTkLabel(window)
        intense1.configure(
            text=f"Nucleus Intensity Sum: {cp.nucleus_intensity[Contour.CONTOUR]}"
        )
        intense2 = customtkinter.CTkLabel(window)
        intense2.configure(text=f"Cellular Intensity Sum: {cp.cell_intensity}")
        intense3 = customtkinter.CTkLabel(window)
        intense3.configure(
            text=f"Line GFP intensity: {cp.get_GFP_line_GFP_intensity()}"
        )
        dist_gfp = customtkinter.CTkLabel(window)
        dist_gfp.configure(text="GFP Distance: {:.3f}".format(cp.gfp_red_dot_distance))
        dist_gfp.grid(row=7, column=5)
        intense1.grid(row=7, column=4)
        intense2.grid(row=8, column=4)
        intense3.grid(row=8, column=5)
        gfp_dot_count_diplay = customtkinter.CTkLabel(window)
        gfp_dot_count_diplay.configure(text=f"gfp count:{cp.get_gfp_dot_count()}")
        gfp_dot_count_diplay.grid(row=9, column=5)

    except Exception:
        print("error with this cell intensity")
    ignore_txt = 'ENABLE' if cp_dict[(image, id)].ignored else 'IGNORE'
    ignore_btn = customtkinter.CTkButton(window, text=ignore_txt, command=partial(ignore, image, id))
    ignore_btn.grid(row=7, column=7, rowspan=2)

    next_btn = customtkinter.CTkButton(window, text="Next Pair", command=partial(display_cell, image, id + 1))
    next_btn.grid(row=10, column=4)
    prev_btn = customtkinter.CTkButton(window, text="Previous Pair", command=partial(display_cell, image, id - 1))
    prev_btn.grid(row=10, column=2)

    # TODO:  Do this in a less stupid way.
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
        prev = list(image_dict.keys())[len(image_dict) - 1]

    image_next_btn = customtkinter.CTkButton(window, text="Next Image", command=partial(display_cell, next, 1))
    image_next_btn.grid(row=4, column=6)
    image_prev_btn = customtkinter.CTkButton(window, text="Previous Image", command=partial(display_cell, prev, 1))
    image_prev_btn.grid(row=4, column=0)
    cp_dict[(image, id)] = cp
    window.update()


def resize_image(image, size_x, size_y, img_format):
    width, height = image.size
    if height > width:
        scale = float(width) / float(height)
        x_scaled = (int(scale * size_x))
        y_scaled = size_y
    else:
        scale = float(height) / float(width)
        x_scaled = size_x
        y_scaled = int(scale * size_y)
    im = image.resize((x_scaled, y_scaled), Image.ANTIALIAS)
    return ImageTk.PhotoImage(im)


def tink(conf, window1):
    global data
    data = conf
    print(data)
    global window
    window = window1

    window.title("Yeast Analysis Tool")
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    # setting tkinter window size
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

    global export_btn
    export_btn = customtkinter.CTkButton(window, text='Export to CSV', command=export_to_csv)
    export_btn.grid(row=0, column=4)
    export_btn['state'] = DISABLED

    global drop_ignored_checkbox
    drop_ignored_checkbox = customtkinter.CTkCheckBox(window, text='drop ignored', variable=drop_ignored)
    drop_ignored_checkbox.grid(row=0, column=5)
    drop_ignored_checkbox['state'] = DISABLED

    distvar = StringVar()

    global input_dir
    input_dir = data['input_dir']
    global output_dir
    output_dir = data['output_dir']

    ignore_btn = customtkinter.CTkButton(window, text="IGNORE")
    ignore_btn.grid(row=6, column=7, rowspan=2)

    global kernel_size_input
    kernel_size_input = data['kernel_size']

    global img_format
    img_format = data['img_format']


    global mcherry_line_width_input
    mcherry_line_width_input = data['mCherry_line_width']

    global kernel_deviation_input
    kernel_deviation_input = data['kernel_diviation']

    global img_title_label
    img_title_label = customtkinter.CTkLabel(window)
    img_title_label.grid(row=3, column=3)

    global img_label
    img_label = customtkinter.CTkLabel(window)
    img_label.grid(row=4, column=1, columnspan=5)

    global ID_label
    ID_label = customtkinter.CTkLabel(window)
    ID_label.grid(row=6, column=0)

    global DIC_label_text
    DIC_label_text = customtkinter.CTkLabel(window)
    DIC_label_text.grid(row=5, column=1)

    global DAPI_label_text
    DAPI_label_text = customtkinter.CTkLabel(window, fg_color='blue')
    DAPI_label_text.grid(row=5, column=2)

    global mCherry_label_text
    mCherry_label_text = customtkinter.CTkLabel(window, fg_color='red')
    mCherry_label_text.grid(row=5, column=3)

    global GFP_label_text
    GFP_label_text = customtkinter.CTkLabel(window, fg_color='green')
    GFP_label_text.grid(row=5, column=4)

    global DIC_label
    DIC_label = customtkinter.CTkLabel(window)
    DIC_label.grid(row=6, column=1)

    global DAPI_label
    DAPI_label = customtkinter.CTkLabel(window)
    DAPI_label.grid(row=6, column=2)

    global mCherry_label
    mCherry_label = customtkinter.CTkLabel(window)
    mCherry_label.grid(row=6, column=3)

    global GFP_label
    GFP_label = customtkinter.CTkLabel(window)
    GFP_label.grid(row=6, column=4)
    img_label.bind("<Button-1>", callback)
    window.bind("<Left>", key)
    window.bind("<Right>", key)
    window.bind("<Up>", key)
    window.bind("<Down>", key)
    for i in range(10):
        window.bind(str(i), key)

    use_cache_var = use_cache.get()
    use_spc110_var = use_spc110.get()
    global image_dict
    image_dict = segment_images.segment_images(data, use_cache_var, use_spc110_var)
    print("test_dict", image_dict)
    if len(image_dict) > 0:
        k, v = list(image_dict.items())[0]
        print("displaycell", k, v[0])
        display_cell(k, v[0])
    window.mainloop()



def callback(event):
    print(f"clicked at {str(event.x)},{str(event.y)}")


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

    if current_image is None or current_cell is None:
        return

    if event.char.isdigit():  # lets you type in integers to go to that cell directly
        keybuf.append(event.char)
        window.after(250, timedout)
        return

    # TODO:  Do this in a less stupid way.
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
        prev = list(image_dict.keys())[len(image_dict) - 1]
    print(f"pressed {repr(event.char)}")
    if event.keysym == 'Left':
        display_cell(current_image, current_cell - 1)
    if event.keysym == 'Right':
        display_cell(current_image, current_cell + 1)
    if event.keysym == 'Up':
        display_cell(prev, 0)
    if event.keysym == 'Down':
        display_cell(next, 0)
