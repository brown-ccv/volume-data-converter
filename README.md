# Volume-data-converter

Project to convert tiff image sequences ( 3D volume ) to a 2D png downscaled texture atlas. This application can be used to port volumes to the [aframe volume viewer](https://github.com/brown-ccv/react-volume-viewer).

## Support to Rhode Island Discovery and Data Center Volume Viewer

You can use this application to query data from an [Erddap server](https://pricaimcit.services.brown.edu/erddap) and convert it into `.raw` and/or `.png` files to visualize it in 3D using the [3D-volume-viewer](https://github.com/brown-ccv/VR-Volumeviewer) and [aframe volume viewer](https://github.com/brown-ccv/react-volume-viewer).

## Prerequisites

- Python > 3.8 or < 3.11
- Pip installed in your environment
- (optional) [Python Poetry](https://github.com/python-poetry/poetry)

## Installation

1. Clone the repo, go to the project's directory.
2. Create a virtual environment using the command  Mac/Unix: `python3 -m venv ./` Windows: `python3 -m venv .\`
3. Activate the virtual environment Mac/Unix: `source venv/bin/activate` Windows: `.\venv\Scripts\activate`
4. Install dependencies:

### Using Python Poetry

- Run `poetry install`  

### Using Pip package managment

- Run `pip install -r requirements.txt`  

The command will install the dependencies for the project.

## Running the available commands

Activate the virtual environment to run the following commands (If you are using poetry you can execute the command by pre-appending the command `poetry run `):

### Converting tiff stack of images to png texture map

To verify the command runs correctly, run the command:

`convert2png --help`

You should see the available parameters that can be used.

- The first mandatory argument is the path to the source folder containing the tiff image sequence.  
- The second mandatory argument is the path to the folder where the png file will be saved.

#### Example

`convert2png path-to\TIFF_Images\ \path-to-result-folder\result.png`

### Converting a raw 3D array to tiff sequence of images

To verify the command runs correctly, run the command:

`raw2tiff --help`

You should see the available paramaters that can be used.

- The first argument is the path to the source folder containing the raw file
- The second argument is the path to the folder the image sequences will be saved
- Third argument is the width of the 3D array
- Fourth argument is the height of the 3D array
- Fifth argument is the depth of the 3D array
  
#### Example

`raw2tiff path-to\raw-file.raw path-to-result-folder 500 550 55`

## FOR OSOM PROJECT - Converting NC files to raw 3D array

### Query data from Erddap

The command `erddap_query` has two arguments:

- `output_file_path`: Path where the result NC file will be saved
- `erddap_configuration_file` (optional): Path to connection configuration file

A configuration file is a `.json` that looks like this:

```json
{
  "erddap_connection": {
    "server": "https://pricaimcit.services.brown.edu/erddap",
    "protocol": "griddap",
    "dataset_id": "model_data_57db_4a85_81d9"
  }, 
  "erddap_constrainsts": {
        "variables":["SalinityBottom","SalinitySurface","WaterTempSurface","WaterTempBottom"],  
        "time>=": "2019-12-30T00:00:00Z",
        "time<=": "2019-12-31T00:00:00Z",
        "time_step": 1,
        "eta_rho>=": 0,
        "eta_rho<=": 1099,
        "eta_rho_step": 1,
        "xi_rho>=": 0,
        "xi_rho<=": 999,
        "xi_rho_step": 1
  }
}
```

Modify the imporant attributes such as `server ULR`, `dataset_id` and `contrainst` according to the data you want to visualize.

There is a default `erddap_configuration.json` file located in the `configuration` folder.

This is an example on how to run the command:

`erddap_query path_to_result_folder path_to_configuration_file.json`

### Converting data from NC to .raw or .png

The command `osom-converter` has only one parameter:

- `parameters_file_path`: Path to a `.json` file that looks like this:

```json
{
  "osom_grid_file": "path_to_grid_file/osom_grid4_mindep_smlp_mod7-netcdf4.nc",
  "osom_data_file": "path_to_nc_datafile/model_data_57db_4a85_81d9.nc",
  "output_folder": "path_to_output_folder",
  "data_descriptor" : "WaterTempSurface",
  "time_frames": [0],
  "layer" : "surface",
  "to_texture_atlas": true
}
```

- There is an example grid file located in the `resources\grid-data` folder
- The `osom_data_file` is the resulting osom-nc file from the `erddap_query` command.
- `data_descriptor` is one of the variables in the erddap data base.
- When the osom-nc files respresent multiple time frames ( i.e data representing different hours in a day ), the data is stored in an 1D-array. The `time_frames` argument specify the time frame you want to convert into volume viewer data. By default, it will convert all the available time frames.
- Some datasets are multi-layered. The `layer` argument lets you specify if the dataset has only one layer (`surface`, `bottom`) or multiple layers (`all`).
- `to_texture_atlas` set to True will convert the nc files into a 2D png texture atlas that can be read by the [aframe volume viewer](https://github.com/brown-ccv/react-volume-viewer).

#### Important for Desktop volume viewer for Mac Users

MacOS users must set the `to_texture_atlas` flag to `True`.

This is an example on how to run the command:

`osom-converter path-to-confiration-file-folder\osom-converter-params.json`
  
## For developers

## Visual code debug

Add the following configuration in your launch.json file.

```json
 configurations": [

        {
            "name": "Python: create_volume_viewer_data test File",
            "type": "python",
            "request": "launch",
            "program": "volume_data_converter/create_volume_viewer_data.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}", 
            "args": ["volume_data_converter/osom-convertert-parameters.json"]
        }

    ]
```