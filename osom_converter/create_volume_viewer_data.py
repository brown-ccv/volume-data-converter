from netCDF4 import Dataset
import netcdftime
import typer
import numpy as np
import os
from osom_converter import scoord_du_new2 as scoor_du
from typing import List

import json
import sys
from pathlib import Path

app = typer.Typer()

constants_file_path = os.path.join(Path(__file__).absolute().parent,"constants.json")

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)

def getVariableValOrDefault(configuration_file, nc_file_handler, variable_name: str):
    if variable_name in nc_file_handler.variables:
        return nc_file_handler.variables[variable_name][:]
    return configuration_file[variable_name]

@app.command()
def createOsomData(
    osom_gridfile: str = typer.Argument(..., help="Grid File with space coordinates"),
    osom_data_file: str = typer.Argument(..., help="NC file osom data"),
    output_folder: str = typer.Argument(
        ..., help="Location where the resuilting .raw files will be saved"
    ),
    data_descriptor: str = typer.Argument(..., help="Descriptor to query the nc file"),
    time_frames: List[int] = typer.Option(
        [],
        help=" List of time frames to convert to raw. By default is None: it will convert all the time frames in a .nc file",
    ),
    layer: str = typer.Option(
        "all",
        help=" Layers of the osom model this nc files maps to. Options: all, surface, bottom",
    )
):

    """
    Converts NETCDF files (*.nc) from the osom model to data files that can be read
    in the volume viewer desktop app
    osom_gridfile: grid file providing uv transformation coordinates ( i.e: file provided by this tool osom_grid4_mindep_smlp_mod7.nc)
    osom_data_file: osom data file with multi-variable data
    output_folder: location where the resulting data will be saved
    data_descriptor: variable to extract from the osom data file (temp, salt)
    """

    try:
        conf_file = open(constants_file_path, 'r')
    except OSError:
        typer.echo(" Could not open/read file: "+ constants_file_path)
        sys.exit()

    with conf_file:
        configuration_file = json.load(conf_file)

    # Default factors/scalers
    verticalLevels = 15
    downscaleFactor = 2

    #check layers
    if layer not in ['all','surface','bottom']:
        raise Exception(f"layer option {layer} not supported")

    # Read files
    typer.echo(" Reading " + osom_gridfile)
    nc_grid_values = Dataset(osom_gridfile, "r")
    typer.echo(" Reading " + osom_data_file)
    nc_dataFile = Dataset(osom_data_file, "r")

    # Assigning grid variables
    h = nc_grid_values.variables["h"][:]
    x_rho = nc_grid_values.variables["x_rho"][:]

    # Assigning data variables
    typer.echo(" Assigning data from data files")
    if data_descriptor not in nc_dataFile.variables:
        raise Exception(f"No variable with name: {data_descriptor}. Options are {nc_dataFile.variables}")

    data = nc_dataFile.variables[data_descriptor][:]
    data_properties =  nc_dataFile.variables[data_descriptor]
    data_dimensions = data_properties.dimensions

    ## zeta needs a different treatment form the other variables. If it's not in the configuration file, then build it from the descriptor's data itself
    if "zeta"  in nc_dataFile.variables:
        zeta = nc_dataFile.variables["zeta"][:]
    else:
        zeta = np.zeros(shape=(data.shape))

     # check if it's a volume or a single slice. Make the corrsponding transformation to 3D array 
    
    if layer == 'all' and "s_rho" not in data_dimensions:
            raise Exception("current dataset does not support elevation layers") 

    elif layer!= 'all' and "s_rho" not in data_dimensions:
        #no elevation data. build a block of 0s    
        new_data = np.ma.zeros((data.shape[0],verticalLevels,data.shape[1],data.shape[2]),dtype=data.dtype)
        data_slice = 0 # bottom by default
        if layer == 'surface':
            data_slice = verticalLevels-1
        
        for i in range(data.shape[0]):
            new_data[i,data_slice,:,:] = data[i,:,:]
        data = new_data


    vtransform = getVariableValOrDefault(configuration_file,nc_dataFile, "Vtransform")
    vstretching = getVariableValOrDefault(configuration_file,nc_dataFile, "Vstretching")
    theta_s = getVariableValOrDefault(configuration_file,nc_dataFile, "theta_s")
    theta_b = getVariableValOrDefault(configuration_file,nc_dataFile, "theta_b")
    hc =getVariableValOrDefault(configuration_file,nc_dataFile, "hc")

    time_variable_name = configuration_file["time_variable"]
    #time_variable_name = "ocean_time"
    if time_variable_name not in nc_dataFile.variables:
        raise Exception("Time variable not found")
    
    ocean_time = nc_dataFile.variables[time_variable_name][:]
    ocean_time_properties = nc_dataFile.variables[time_variable_name]

    # Compute and plot ROMS vertical stretched coordinates
    typer.echo(" Computing ROMS vertical stretched coordinates")
    z, s, C = scoor_du.scoord_du_new2(
        h,
        zeta[0, :, :],
        vtransform,
        vstretching,
        theta_s,
        theta_b,
        hc,
        verticalLevels,
        0,
    )

    min_z = np.floor(z.min())
    max_z = np.ceil(z.max())

    # set nbSlices (per meter)
    query_depths = np.arange(min_z, max_z + 1)
    query_depths_shape = np.shape(query_depths)
    slices = query_depths_shape[0]

    # getting minimum and max data
    min_data = np.floor(data.min())
    max_data = np.ceil(data.max())

    # scale data to the range of 0 and 1
    data = (data - min_data) / (max_data - min_data)

    OsomDataFilePath, osom_data_filename = os.path.split(osom_data_file)
    osom_data_filename, ext = os.path.splitext(osom_data_filename)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if len(time_frames) == 0:
        time_frames = np.arange(np.shape(data)[0]).tolist()

    typer.echo(" Converting nc data to raw and desc files. This process will take time")
    with typer.progressbar(time_frames, label="Processing Time") as time_progress:
        for time_t in time_progress:
            # intiialize with zero
            out_data = np.zeros((slices, np.shape(x_rho)[0], np.shape(x_rho)[1]))
            progress_bar_label = "Processing Time Step " + str(time_t)
            with typer.progressbar(
                range(np.shape(x_rho)[0]), label=progress_bar_label
            ) as gridx_progress:
                for x in gridx_progress:
                    for y in range(np.shape(x_rho)[1]):
                        # get depth column
                        t_data = data[time_t, :, x, y]
                        t_data = np.ma.squeeze(t_data)
                        idx = ~t_data.mask
                        
                        if np.sum(idx.astype(int)) > 0:
                            # read set depth
                            idx_1d = np.atleast_1d(idx)
                            domain = z[idx_1d, x, y]
                            # read values
                            mapped_values = np.ma.round(t_data[idx], decimals=4)
                            t_new = np.interp(
                                query_depths, domain, mapped_values, left=0, right=0
                            )
                            out_data[:, x, y] = np.round(t_new, decimals=4)

            ## downscale data
            out_data = out_data[::downscaleFactor, ::downscaleFactor, ::downscaleFactor]
            out_dataShape = out_data.shape
            out_data = out_data.reshape(
                (out_dataShape[2], out_dataShape[1], out_dataShape[0])
            )

            digits = len(str(np.shape(data)[0] + 1))
            ## save data file
            data_filename = (
                f"{data_descriptor}_{osom_data_filename}_timestep{time_t:0{digits}}"
            )

            with open(
                os.path.join(output_folder, data_filename + ".raw"), "wb"
            ) as data_file:
                outData32 = out_data.astype(np.float32)
                outData32.tofile(data_file)

            ## save description file
            with open(
                os.path.join(output_folder, data_filename + ".desc"), "w"
            ) as desc_file:
                fprintf(
                    desc_file,
                    "%u,%u,%u,%.6f,%.6f\n",
                    np.shape(out_data)[0],
                    np.shape(out_data)[1],
                    np.shape(out_data)[2],
                    min_data,
                    max_data,
                )
                ocean_times = netcdftime.num2date(
                    ocean_time[time_t],
                    units=ocean_time_properties.units,
                    calendar=ocean_time_properties.calendar,
                )
                fprintf(desc_file, "%s\n", ocean_times.strftime("%Y-%m-%d %H:%M:%S"))
    typer.echo(" End of process")

if __name__ == "__main__":
    app()


def main():
    app()