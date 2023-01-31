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
