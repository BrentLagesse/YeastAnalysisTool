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

!!! Important !!!<br/>
If pulled a different branch other than ryota-main, code might not run and will need slight changes to run on windows. <br/>

Change foreground parameters to fg_color<br/>
Change text_font parameters to font<br/>

In main.py: <br/>
Look for<br/>
&emspcsvwriter = csv.writer(csvfile)<br/>
Change to <br/>
&emspcsvwriter = csv.writer(csvfile, lineterminator='\n')<br/>

image.astype(np.float32)
becomes, 
&emspnp.float32(image)
