# tiff-volume-2png

Project to convert tiff image sequences ( 3D volume ) to a 2D png downscaled file. This application can be used to port volumes to the [aframe volume viewer](https://github.com/brown-ccv/react-volume-viewer).

## Prerequisites

- Make sure python 3.8 or greater is installed in your system
- [Python Poetry](https://github.com/python-poetry/poetry)

## Installation

Clone the repo, go to the project's directory and run poetry to install all dependencies:

`poetry install`  

The command will install the dependencies for the project.

## Converting tiff sequence of images to png texture map

Run the command:

`poetry run convert2png --help`

To verify the command runs correctly. You should see the available paramaters that can be used.

  - The first mandatory argument is the path to the source folder containing the tiff image sequence.  
  - The second mandatory argument is the path to the folder where the png file will be saved.

### Example

`poetry run convert2png path-to\TIFF_Images\ \path-to-result-folder\result.png`

## Converting a raw 3D array to tiff sequence of images

Run the command:

`poetry run raw2tiff --help`

To verify the command runs correctly. You should see the available paramaters that can be used.

  - The first argument is the path to the source folder containing the raw file. 
  - The second argument is the path to the folder the image sequences will be saved. 
  - Third argument is the width of the 3D array
  - Fourth argument is the height of the 3D array
  - Fifth argument is the depth of the 3D array
  
### Example

`poetry run raw2tiff path-to\raw-file.raw path-to-result-folder 500 550 55`


## FOR OSOM PROJECT - Converting NC files to raw 3D array

Run the command:

`poetry run osom-converter --help`

To verify the command runs correctly. You should see the available paramaters that can be used.

  - The first argument is the path to the source Gird file (There an example in this repo)
  - The second argument is the path to the osom NC file.
  - Third argument is the path to the output_folder of the raw file
  - Fourth argument is the attribute in the nc file to be analyzed
  - (Optional) Fifth argument is a list of time frames to be analyzed (by default analyzes the whole dataset)
  
 ### Example

`poetry run osom-converter path-to-osom-grid-file\osom_grid4_mindep_smlp_mod7.nc path-to-osom-nc-file\ocean_his_0196.nc path-to-output-folder temp`
  
### Example

`poetry run osom-converter path-to\nc-file.raw path-to-result-folder`

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
 
