# YeastAnalysisTool

For usage on a Mac, do the following.


git clone https://github.com/BrentLagesse/YeastAnalysisTool.git
cd YeastAnalysisTool
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.mac.txt
mkdir weights
mkdir outputs
mkdir outputs/segmented

Then -- Download this link https://drive.google.com/file/d/1moUKvWFYQoWg0z63F0JcSd3WaEPa4UY7/view?usp=sharing and put it in the weights directory you just made.

python3 main.py 

This starts the program.  Change the Input directory to one of your picture directories like M2210

click start analysis and it should start going (the button will probably turn blue and stay that way a while.  If it doesn't, click it again -- this is a weird bug I'm finding on the Mac that doesn't happen on linux and I haven't had time to fix it yet)

After that runs (it'll take a minute or so, probably), you should see the images and the cells.  
