# YeastAnalysisTool

You need to make sure git, virtualenv, and python3 are installed and are in the $PATH (you can type those command names on the commandline and your computer finds them).



For usage on a Mac, do the following in a terminal.


git clone https://github.com/BrentLagesse/YeastAnalysisTool.git

cd YeastAnalysisTool

virtualenv venv

source venv/bin/activate

sh init.sh

#NOTE:  If you are using an M1 mac, run sh initm1.sh instead.  

You can do this while the other script is still running -- Download this link https://drive.google.com/file/d/1moUKvWFYQoWg0z63F0JcSd3WaEPa4UY7/view?usp=sharing and put it in the weights directory the previous script just made.

python3 main.py 

This starts the program.  Change the Input directory to one of your picture directories like M2210.  See Naming Scheme

click start analysis and it should start going (the button will probably turn blue and stay that way a while. 

After that runs (it'll take a few minutes or so, probably), you should see the images and the cells.  

# Naming Scheme

The system currently follows a strict naming scheme though we hope to make this more flexible in the future.

In the main directory that you load in, you have a lot of files like this:

2021_0902_M2351_    --- This is the BaseName of the files and can be anything
This is immediately followed by a IdNumber like 001 (so if you had 8 images, you'd have 001 through 008)
Then you have:  R3D_REF.tif

So overall it looks like:   2021_0902_M2351_008_R3D_REF.tif

Also in the main directory are subdirectories and they look like this -- 2021_0902_M2351_008_PRJ_TIFFS which is just the BaseName + IdNumber + _PRJ_TIFFS
and that holds the additional images for each main image. 

Those images should be named something like

2021_0902_M2351_008_PRJ_w435.tif

and there are 3 Type  --   w435 (DAPI), w525 (GFP), and w625 (mCherry) -- At one point we had support for CFP but it is not currently being used.

which is just BaseName + IdNumber + _PRJ_ + Type + .tif

