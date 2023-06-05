from cgitb import enable, text
from distutils.cmd import Command
from multiprocessing.sharedctypes import Value
import shutil
import tkinter
import customtkinter
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

import opts as opt
import json
import main
import os

input_dir = opt.input_directory
output_dir = opt.output_directory

current_image = None
current_cell = None

configure_file = open("./config.json")
global data
data = json.load(configure_file)
data['input_dir'] = input_dir
data['output_dir'] = output_dir


def on_resize(event):
    try:
        w = width
        h = height
    except Exception:
        return


def refresh(self):
    self.destroy()
    self.__init__()


def set_input_directory():
    global input_dir
    old = input_dir
    global input_lbl
    input_dir = filedialog.askdirectory(parent=configure_window, title='Choose the Directory with the input Images',
                                        initialdir=input_dir)
    if input_dir == "":
        input_dir = old
        return
    input_lbl.configure(text=input_dir)
    data['input_dir'] = input_dir


def set_output_directory():
    global output_dir
    old = output_dir
    global output_lbl
    output_dir = filedialog.askdirectory(parent=configure_window,
                                         title='Choose the Directory to output Segmented Images',
                                         initialdir=output_dir)
    if output_dir == "":
        output_dir = old
        return
    output_lbl.configure(text=output_dir)
    data['output_dir'] = output_dir


def set_data():
    data['mCherry_line_width'] = mCherry_var.get()
    data['kernel_size'] = kernel_size_var.get()
    data['kernel_diviation'] = kernel_deviation_var.get()
    data['useChache'] = cache_var.get()
    data['mCherry_to_find_pairs'] = mCherry_toggle_var.get()
    data['drop_ignore'] = ignore_var.get()


def get_data():
    mCherry_var.set(data['mCherry_line_width'])
    kernel_size_var.set(data['kernel_size'])
    kernel_deviation_var.set(data['kernel_diviation'])
    cache_var.set(data['useChache'])
    mCherry_toggle_var.set(data['mCherry_to_find_pairs'])
    ignore_var.set(data['drop_ignore'])
    input_dir = data['input_dir']
    output_dir = data['output_dir']
    img_format = data['img_format']
    output_lbl.configure(text=output_dir)
    input_lbl.configure(text=input_dir)


def set_ok():
    start_btn.configure(state="normal")
    set_data()
    # with open('pre_config.json', 'w') as fp:
    #     json.dump(data, fp)
    configure_window.destroy()


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

window = customtkinter.CTk()
window.title("Yeast Analysis Tool")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
# setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.bind("<Configure>", on_resize)


def start_analysis():
    if os.path.exists("./pre_config.json"):
        preconf = open("./pre_config.json")
        preconf_data = json.load(preconf)
        print(preconf_data['input_dir'], data['input_dir'])
        if preconf_data['input_dir'] != data['input_dir']:
            path = data['output_dir'] + '/*'
            print(path)
            if os.path.exists(data['output_dir'] + '/segmented'):
                shutil.rmtree(data['output_dir'] + '/segmented')
            if os.path.exists(data['output_dir'] + '/masks'):
                shutil.rmtree(data['output_dir'] + '/masks')

    if not os.path.isdir(data['output_dir'] + "/segmented/"):
        os.makedirs(data['output_dir'] + "/segmented/")
    if not os.path.isdir(data['output_dir'] + "/masks/"):
        os.makedirs(data['output_dir'] + "/masks/")
    with open('pre_config.json', 'w') as fp:
        json.dump(data, fp)
    main.tink(data, window)


def set_default():
    set_data()
    with open('default_config.json', 'w') as fp:
        json.dump(data, fp)
    configure_window.destroy()


def my_configure():
    configure_file = open("./default_config.json")
    global data
    data = json.load(configure_file)
    get_data()


def pre_conf():
    configure_file = open("./pre_config.json")
    global data
    data = json.load(configure_file)
    get_data()


def button_function():
    print("hello")


cache_var = customtkinter.StringVar(value=data['useChache'])
print(cache_var.get())


def cache_switch_event():
    print("switch toggled, current value:", cache_var.get())


mCherry_toggle_var = customtkinter.StringVar(value=data['mCherry_to_find_pairs'])


def mCherry_switch_event():
    print("switch toggled, current value:", mCherry_toggle_var.get())


ignore_var = customtkinter.StringVar(value=data['drop_ignore'])


def ignore_switch_event():
    print("switch toggled, current value:", ignore_var.get())


optionmenu_var = customtkinter.StringVar(value="Metaphase Arrested")  # set initial value


def optionmenu_callback(choice):
    print("optionmenu dropdown clicked:", choice)
    data['arrested'] = choice

def img_optionmenu_callback(choice):
    print("optionmenu dropdown clicked:", choice)
    data['img_format'] = choice


kernel_size_var = tkinter.StringVar(value=data['kernel_size'])
mCherry_var = tkinter.StringVar(value=data['mCherry_line_width'])
kernel_deviation_var = tkinter.StringVar(value=data['kernel_diviation'])


def configuration_window():
    global configure_window
    configure_window = customtkinter.CTkToplevel(window)
    configure_window.geometry("450x500")
    configure_window.wm_transient(window)

    # input directory button
    input_btn = customtkinter.CTkButton(configure_window, text="Set Input Directory", command=set_input_directory)
    input_btn.grid(row=0, column=0, padx=10, pady=10)

    global input_lbl
    input_lbl = customtkinter.CTkLabel(configure_window, text=input_dir)
    input_lbl.grid(row=0, column=1, padx=10, pady=10)

    # output directory button
    output_btn = customtkinter.CTkButton(configure_window, text="Set Output Directory", command=set_output_directory)
    output_btn.grid(row=1, column=0, padx=10, pady=10)

    global output_lbl
    output_lbl = customtkinter.CTkLabel(configure_window, text=output_dir)
    output_lbl.grid(row=1, column=1, padx=10, pady=10)


    kernel_label = customtkinter.CTkLabel(master=configure_window, text="Kernel size", )
    kernel_label.grid(row=2, column=0, padx=10, pady=10, ipadx=0)

    kernel_size_input = customtkinter.CTkEntry(master=configure_window, placeholder_text="Keral size",
                                               textvariable=kernel_size_var)
    kernel_size_input.grid(row=2, column=1, padx=10, pady=10)

    kernel_deviation_label = customtkinter.CTkLabel(master=configure_window, text="Kernel deviation")
    kernel_deviation_label.grid(row=3, column=0, padx=10, pady=10)

    kernel_deviation_input = customtkinter.CTkEntry(master=configure_window, placeholder_text="Keral size",
                                                    textvariable=kernel_deviation_var)
    kernel_deviation_input.grid(row=3, column=1, padx=10, pady=10)

    mCherry_label = customtkinter.CTkLabel(master=configure_window, text="mCherry line width")
    mCherry_label.grid(row=4, column=0, padx=10, pady=10, ipadx=0)

    mCherry_input = customtkinter.CTkEntry(master=configure_window, placeholder_text="Keral size",
                                           textvariable=mCherry_var)
    mCherry_input.grid(row=4, column=1, padx=10, pady=10)

    # cache switch
    cache = customtkinter.CTkSwitch(master=configure_window, text="Cache", command=cache_switch_event,
                                    variable=cache_var, onvalue="on", offvalue="off")
    cache.grid(row=5, column=0, padx=10, pady=10)

    # mCherry switch
    mCherry_toggle = customtkinter.CTkSwitch(master=configure_window, text="mCherry to find pairs",
                                             command=mCherry_switch_event,
                                             variable=mCherry_toggle_var, onvalue='on', offvalue='off')
    mCherry_toggle.grid(row=5, column=1, padx=10, pady=10)

    # cache switch
    ignore = customtkinter.CTkSwitch(master=configure_window, text="drop ignore", command=ignore_switch_event,
                                     variable=ignore_var, onvalue='on', offvalue='off')
    ignore.grid(row=6, column=0, padx=10, pady=10)

    # choice dropdown
    combobox = customtkinter.CTkComboBox(master=configure_window,
                                          values=["Metaphase Arrested", "G1 Arrested"],
                                         command=optionmenu_callback,
                                         variable=data['arrested'])
    combobox.grid(row=6, column=1, padx=10, pady=10)

    image_format_label = customtkinter.CTkLabel(master=configure_window, text="Image Format", )
    image_format_label.grid(row=7, column=0, padx=10, pady=10, ipadx=0)

    image_format_input = customtkinter.CTkComboBox(master=configure_window,
                                          values=["tiff", "DV"],
                                         command=img_optionmenu_callback,
                                         variable=data['img_format'])
    image_format_input.grid(row=7, column=1, padx=10, pady=10)

    # My_configurations button
    my_btn = customtkinter.CTkButton(configure_window, text="Get my Configuration", command=my_configure,
                                     fg_color="green", hover=True)
    my_btn.grid(row=8, column=0, padx=10, pady=10)

    # set_default button
    set_btn = customtkinter.CTkButton(configure_window, text="Set as default", command=set_default, fg_color="green",
                                      hover=True)
    set_btn.grid(row=8, column=1, padx=10, pady=10)

    # Ok button
    ok_btn = customtkinter.CTkButton(configure_window, text="ok", command=set_ok, fg_color="green", hover=True)
    ok_btn.grid(row=9, column=0, padx=10, pady=10)

    # Ok button
    ok_btn = customtkinter.CTkButton(configure_window, text="Previous Configuration", command=pre_conf,
                                     fg_color="green", hover=True)
    ok_btn.grid(row=9, column=1, padx=10, pady=10)


btn = customtkinter.CTkButton(window, text="Configure", command=configuration_window)
btn.grid(row=0, column=0, padx=10, pady=10)

export_btn = customtkinter.CTkButton(window, text='Export to CSV', command=button_function, state="disabled", )
export_btn.grid(row=0, column=1, padx=10, pady=10)

start_btn = customtkinter.CTkButton(window, text='start analysis', command=start_analysis, state="disabled", )
start_btn.grid(row=0, column=2, padx=10, pady=10)

window.mainloop()
