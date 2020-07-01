# Simple trained GMM voice conversion model
## Usage
#### command line version
Assuming we have a trained model in the same directory, i.e. **class.pkl** in this repo. If one wants to convert voice from file **1.wav**, put it in **src** folder and run this following command
```
python cmd_vc.py -m class.pkl -f 1.wav
```
Then output file will be in **tgt** folder.

If one want to convert many wav files, place them in a folder i.e. **input**, and run 
```
python cmd_vc.py -m class.pkl -fd input
```
A folder name **tgt_input** containing converted wav files will appear.
#### GUI version
Put directory to media player on your PC on **player.txt**, for example it is C:\Program Files (x86)\K-Lite Codec Pack\MPC-HC64\mpc-hc64.exe on my PC. Compile vc.py.
