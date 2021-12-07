# tiff-volume-2png

Project to convert tiff image sequences ( 3D volume ) to a 2D png downscaled file. This application can be used to port volumes to the aframe volume viewer.

## Prerequisites

- Make sure python 3.6 or greated is installed in your system
- PIP 19.8 or greater
- Clone the repo and create a virtual enviroment in the same directory level

## Installation

To install all dependencies run:
`pip install -r requirements.txt`  

## Execute

The main script is `tiff-2-png-converter.py`.  It has multiple flags you can run the script with. 
To get help, on the command line run:

`python tiff-2-png-converter.py --help`

The flag `-s` is the path to the source file containing the tiff image sequence, and `-d` the path to the folder where the png file will be saved.

`python tiff-2-png-converter.py -s C:\\tiff-volume-2png\\TIFF_Images\\ -d C:\\tiff-volume-2png\\result.png`


# For developers

## Visual code debug

Add the following configuration in your launch.json file.

 `
 "name": "Python: Current File",`  <br/>
 `"type": "python",  ` <br/>
 `"request": "launch",  ` <br/>
 `"program": "tiff-2-png-converter.py",  ` <br/>
 `"console": "integratedTerminal",  ` <br/>
 `"cwd": "C:\\tiff-volume-2png",  ` <br/>
 `"args": ["--dir","D:\\tiff-volume-2png\\tiffs\\"]`  <br/>
 
