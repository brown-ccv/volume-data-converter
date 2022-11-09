from re import L
from netCDF4 import Dataset
import cftime
import typer
import numpy as np
import os
from volume_data_converter import scoord_du_new2 as scoor_du
from typing import List
import shutil

import json
import sys
from pathlib import Path

app = typer.Typer()


def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)


@app.command()
def create_osom_data(
    parameters_file_path: str = typer.Argument(
        ..., help="Path to parameters json file"
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

    with open(parameters_file_path, "r") as parameters_json:
        parameters = json.load(parameters_json)

    osom_gridfile = parameters["osom_grid_file"]
    osom_data_file = parameters["osom_data_file"]
    output_folder = parameters["output_folder"]
    data_descriptor = parameters["data_descriptor"]
    time_frames = parameters.get("time_frames",None)
    layer =  parameters.get("layer","all")
    
    osom_constants_file_path = os.path.join(
        Path(__file__).absolute().parent, "config", "constants.json"
    )

    resources_folder_path = os.path.join(Path(__file__).absolute().parent, "resources","volume-viewer")

    osom_const_file = open(osom_constants_file_path, "r")
    osom_configuration_dicc = json.load(osom_const_file)

    # Default factors/scalers
    verticalLevels = osom_configuration_dicc.get("verticalLevels", 15)
    downscaleFactor = osom_configuration_dicc.get("downscaleFactor", 2)

    # check layers
    if layer not in ["all", "surface", "bottom"]:
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
        raise Exception(
            f"No variable with name: {data_descriptor}. Options are {nc_dataFile.variables}"
        )

    data = nc_dataFile.variables[data_descriptor][:]
    data_properties = nc_dataFile.variables[data_descriptor]
    data_dimensions = data_properties.dimensions

    ## zeta needs a different treatment form the other variables. If it's not in the configuration file, then build it from the descriptor's data itself
    if "zeta" in nc_dataFile.variables:
        zeta = nc_dataFile.variables["zeta"][:]
    else:
        zeta = np.zeros(shape=data.shape)

    # check if it's a volume or a single slice. Make the corrsponding transformation to 3D array

    if layer.lower() == "all":
        # s_rho dimension implies that the data is distributed on multiple layers
        # all is the default value. Check if the dataset is multilayer or not.
        if "s_rho" not in data_dimensions:
            raise Exception("current dataset does not support elevation layers")
    else:
        # no elevation data. Map the data to a empty block of 'verticalayers' layers.
        # Our current cases are surface and bottom. They map to layer 0 and 14 respectively
        new_data = np.ma.zeros(
            (data.shape[0], verticalLevels, data.shape[1], data.shape[2]),
            dtype=data.dtype,
        )
        data_slice = 0  # bottom by default
        if layer == "surface":
            data_slice = -1

        for i in range(data.shape[0]):
            new_data[i, data_slice, :, :] = data[i, :, :]
        data = new_data

    vtransform = nc_dataFile.variables.get(
        "Vtransform", osom_configuration_dicc["Vtransform"]
    )
    vstretching = nc_dataFile.variables.get(
        "Vstretching", osom_configuration_dicc["Vstretching"]
    )
    theta_s = nc_dataFile.variables.get("theta_s", osom_configuration_dicc["theta_s"])
    theta_b = nc_dataFile.variables.get("theta_b", osom_configuration_dicc["theta_b"])
    hc = nc_dataFile.variables.get("hc", osom_configuration_dicc["hc"])

    time_variable_name = osom_configuration_dicc["time_variable"]
    if time_variable_name not in nc_dataFile.variables:
        raise Exception("Time variable not found")

    ocean_time_header = nc_dataFile.variables[time_variable_name]
    ocean_time = ocean_time_header[:]

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
    data -= min_data
    data /= max_data - min_data

    OsomDataFilePath, osom_data_filename = os.path.split(osom_data_file)
    osom_data_filename, ext = os.path.splitext(osom_data_filename)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # create volume-viewer osom-data output folder
    output_folder = os.path.join(output_folder, f"osom-data-{data_descriptor}")
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

                        if any(idx):
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

            output_data_folder = os.path.join(output_folder, "data")
            if not os.path.exists(output_data_folder):
                os.mkdir(output_data_folder)

            with open(
                os.path.join(output_data_folder, data_filename + ".raw"), "wb"
            ) as data_file:
                outData32 = out_data.astype(np.float32)
                outData32.tofile(data_file)

            ## save description file
            desc_file_path = os.path.join(output_data_folder, data_filename + ".desc")
            with open(desc_file_path, "w") as desc_file:
                fprintf(
                    desc_file,
                    "%u,%u,%u,%.6f,%.6f\n",
                    np.shape(out_data)[0],
                    np.shape(out_data)[1],
                    np.shape(out_data)[2],
                    min_data,
                    max_data,
                )
                ocean_times = cftime.num2date(
                    ocean_time[time_t],
                    units=ocean_time_header.units,
                    calendar=ocean_time_header.calendar,
                )
                fprintf(desc_file, "%s\n", ocean_times.strftime("%Y-%m-%d %H:%M:%S"))
    typer.echo(" Creating volume viewer package")
    shutil.copytree(resources_folder_path, output_folder, dirs_exist_ok=True)
    with open(os.path.join(output_folder, "osom-loader.txt"), "a") as loader_file:
        fprintf(loader_file, f"numVolumes 1 {data_descriptor} \n")
        fprintf(
            loader_file, f"volume1  data/{data_filename}.desc 1 1 1 0 0 0 raycast 1\n"
        )

    typer.echo(" End of process")


def main():
    app()


if __name__ == "__main__":
    main()
