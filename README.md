# YeastAnalysisTool

Note --  This project is in the process of being deprecated.  We are currently testing a web-based version that is easier to install in your lab.  Future updates will be found in that version and I will link to it here when it is publicly available.

You need to make sure git, virtualenv, and python3 are installed and are in the $PATH (you can type those command names on the commandline and your computer finds them).

Note that I have mostly used python 3.9.6 for mac.  It seems that 3.12.2 may be causing problems with the dependencies.



For usage on a Mac, do the following in a terminal.


git clone https://github.com/BrentLagesse/YeastAnalysisTool.git

cd YeastAnalysisTool

virtualenv venv

source venv/bin/activate

sh init.sh

#NOTE:  If you are using an M1 mac, run sh initm1.sh instead.  

You can do this while the other script is still running -- Download this link https://drive.google.com/file/d/1moUKvWFYQoWg0z63F0JcSd3WaEPa4UY7/view?usp=sharing and put it in the weights directory the previous script just made.

python3 display.py 

This starts the program.  Change the Input directory to one of your picture directories like M2210

click start analysis and it should start going (the button will probably turn blue and stay that way a while. 

After that runs (it'll take a few minutes or so, probably), you should see the images and the cells.  

###########################################################################################################################################################

Running on Windows
Having the same assumptions that Python(3.10) is installed in the machine

1. Git clone https://github.com/BrentLagesse/YeastAnalysisTool.git  #Clone github Repo using

2. cd YeastAnalysisTool

3. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py          #Download pip

4. python get-pip.py

5. py -m pip install --upgrade pip

6. py -m pip install --user virtualenv                              #install virtual environment

7. py -m venv venv                                                  #create venv folder

8. .\venv\Scripts\activate                                          #activate venv

9. pip install -r requirementsWindows.txt                           #install all windows requirements

10. Run init.bat

11. Download this file https://drive.google.com/file/d/1moUKvWFYQoWg0z63F0JcSd3WaEPa4UY7/view
Save the downloaded .h5 file to the “.\weights” directory created by running init.bat (for windows)
Create output Folders within the YeastAnalysisTool folder, and create a folder:  “.\output\segmented”

12. 
!!! Important !!!<br/>
Windows and Mac might have some conflicting issues, so follow the terminal and fix those issues. Some issues below for windows maybe: <br/>

Change foreground parameters to fg_color<br/>
Change text_font parameters to font<br/>

In main.py: <br/>
Look for<br/>
&emsp;csvwriter = csv.writer(csvfile)<br/>
Change to <br/>
&emsp;csvwriter = csv.writer(csvfile, lineterminator='\n')<br/>

image.astype(np.float32)<br/>
becomes,<br/>
&emsp;np.float32(image)

## Additional Instructions
1. Make sure the name of the file is [NAME OF THE PICTURE]_001_PRJ.dv and the input dir matches the name of all the file's [NAME OF THE PICTURE] Ex. M2067_001_PRJ.dv file would be inside a M2067 folder.
2. The output dir location doesn't matter
3. due to different os, read the errors, it might not work out of the gate and need to edit some lines like the line above about fonts
4. The code works about 70% of the time. Try with cache on and off
5. <b> Due to the machine learning part only works on certain versions of packages, we have to specifically use them <b> the easiest way do to do is to delete all your personal pip packages and reinstall them. Please follow instructions below

```bash
# puts all personal packages into deleteRequirements.txt
1. pip freeze --all > deleteRequirements.txt
# uninstalls all packages
2. pip uninstall -r deleteRequirements.txt
# installs repo's pip packages
3. pip install -r ./requirements.txt --no-cache-dir
#deletes temporary Requirements
4. del deleteRequirements.txt
```
