# tiff-volume-2png

Project to convert tiff image sequences ( 3D volume ) to a 2D png downscaled file. This application can be used to port volumes to the [aframe volume viewer](https://github.com/brown-ccv/react-volume-viewer).

## Prerequisites

- Make sure python 3.6 or greater is installed in your system
- PIP 19.8 or greater
- [Python Poetry](https://github.com/python-poetry/poetry)

## Installation

Clone the repo, go to the project's directory and run poetry to install all dependencies:

`poetry install`  

The command will install the dependencies for the project.

## Execute

Run the command:

`poetry run convert2png --help`

To verify the application runs correctly. You should see the available paramaters that can be used with the command convert2png.

The first mandatory argument is the path to the source file containing the tiff image sequence. The second mandatory argument is the path to the folder where the png file will be saved.

`python run convert2png \you-path-to\TIFF_Images\ \tiff-volume-2png\result.png`


# For developers

## Visual code debug

Add the following configuration in your launch.json file.

 `
 "name": "Python: Current File",`  <br/>
 `"type": "python",  ` <br/>
 `"request": "launch",  ` <br/>
 `"program": "tiff-2-png-converter.py",  ` <br/>
 `"console": "integratedTerminal",  ` <br/>
 `"cwd": "tiff-volume-2png",  ` <br/>
 `"args": ["\you-path-to\TIFF_Images\","\tiff-volume-2png\result.png"]`  <br/>
 
