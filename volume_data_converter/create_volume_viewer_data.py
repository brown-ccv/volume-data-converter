from re import L
from netCDF4 import Dataset
import cftime
import typer
import numpy as np
import os
from volume_data_converter import scoord_du_new2 as scoor_du
from typing import List
from distutils.dir_util import copy_tree

import json
import sys
from pathlib import Path
from PIL import Image
from volume_data_converter import tiff_2_png_converter

app = typer.Typer()


def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)


@app.command()
def create_osom_data(
    parameters_file_path: str = typer.Argument(..., help="Path to parameters json file")
):

    """
    Converts NETCDF files (*.nc) from the osom model to data files that can be read
    in the volume viewer desktop app
    parameters_file_path: Path to parameters json file
    Parameter descriptor file is a Json text file that must contain the following key-value attributes:
    osom_data_file(str): osom data file with multi-variable data
    output_folder(str): location where the resulting data will be saved
    data_descriptor(str): variable to extract from the osom data file (temp, salt)
    time_frames(array): if the nc dataset contains multiple time frames, this array specifies the frames to convert to volume viewer data. By default it will convert all the dataset.
    layer(str): options are all(defautl),surface or bottom
    to_texture_atlas(bool): export nc dataset as a 2D texture atlas instead of 3D volume matrix
    """

    with open(parameters_file_path, "r") as parameters_json:
        parameters = json.load(parameters_json)

    osom_gridfile = parameters["osom_grid_file"]
    osom_data_file = parameters["osom_data_file"]
    output_folder = parameters["output_folder"]
    data_descriptor = parameters["data_descriptor"]
    time_frames = parameters.get("time_frames", None)
    layer = parameters.get("layer", "all")
    to_texture_atlas = parameters.get("to_texture_atlas", False)

    osom_constants_file_path = os.path.join(
        Path(__file__).absolute().parent, "config", "constants.json"
    )

    resources_folder_path = os.path.join(
        Path(__file__).absolute().parent, "resources", "volume-viewer"
    )

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
    single_layer_data = False
    if layer.lower() == "all":
        # s_rho dimension implies that the data is distributed on multiple layers
        # all is the default value. Check if the dataset is multilayer or not.
        if "s_rho" not in data_dimensions:
            raise Exception("current dataset does not support elevation layers")
    else:
        # Single layer case (bottom, surface). Create 3D matrix for each time frame with the
        # current data and mask the non-relevant ( no data ) slices.
        data = np.ma.resize(
            data, (data.shape[0], verticalLevels, data.shape[1], data.shape[2])
        )
        data[:, 1:, :, :].mask = True
        single_layer_data = True

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

    # create top level volume-viewer osom-data output folder
    top_level_output_folder = os.path.join(
        output_folder, f"osom-data-{data_descriptor}"
    )
    if not os.path.exists(top_level_output_folder):
        os.mkdir(top_level_output_folder)

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

                            if not single_layer_data:
                                t_new = np.interp(
                                    query_depths, domain, mapped_values, left=0, right=0
                                )
                                out_data[:, x, y] = np.round(t_new[0:4], decimals=4)
                            else:
                                t_new = np.interp(query_depths, domain, mapped_values)
                                if layer == "surface":
                                    out_data[-20:, x, y] = np.round(
                                        t_new[0:20], decimals=4
                                    )
                                else:
                                    # bottom
                                    out_data[:20, x, y] = np.round(
                                        t_new[0:20], decimals=4
                                    )

            ## downscale data
            out_data = out_data[::downscaleFactor, ::downscaleFactor, ::downscaleFactor]

            ## set up result data file name path and extensions
            digits = len(str(np.shape(data)[0] + 1))
            output_data_folder = os.path.join(top_level_output_folder, "data")
            if not os.path.exists(output_data_folder):
                os.mkdir(output_data_folder)

            data_filename = (
                f"{data_descriptor}_{osom_data_filename}_timestep{time_t:0{digits}}"
            )
            ocean_times = cftime.num2date(
                ocean_time[time_t],
                units=ocean_time_header.units,
                calendar=ocean_time_header.calendar,
            )
            if not to_texture_atlas:
                ## save to regular 3D volume

                save_file_path = os.path.join(
                    output_data_folder, data_filename + ".raw"
                )
                save_raw_data(out_data, digits, save_file_path)
                desc_file_path = os.path.join(
                    output_data_folder, data_filename + ".desc"
                )
                out_data_shape = np.shape(out_data)
                volume_spacing = "1 1 1 0 0 0"
                save_description_file(
                    desc_file_path,
                    out_data_shape[0],
                    out_data_shape[1],
                    out_data_shape[2],
                    min_data,
                    max_data,
                    ocean_times,
                )
            else:
                ## save to 2D texture atlas

                data_file_path = os.path.join(
                    output_data_folder, data_filename + ".png"
                )
                img_width, img_height, num_slices = save_texture_atlas(
                    out_data, data_file_path
                )

                ## save description file
                volume_spacing = "2 2 2 0 0 0"
                desc_file_path = os.path.join(
                    output_data_folder, data_filename + ".png.desc"
                )
                save_description_file(
                    desc_file_path,
                    img_width,
                    img_height,
                    num_slices,
                    min_data,
                    max_data,
                    ocean_times,
                )

    typer.echo(" Creating volume viewer package")

    copy_tree(resources_folder_path, top_level_output_folder)
    with open(
        os.path.join(top_level_output_folder, "osom-loader.txt"), "a"
    ) as loader_file:
        fprintf(loader_file, f"numVolumes 1 {data_descriptor} \n")
        fprintf(
            loader_file,
            f"volume1  data/{data_filename}.desc {volume_spacing} raycast 1\n",
        )

    typer.echo(" End of process")


def save_texture_atlas(volue_data: np.array, save_file_path: str):
    num_slices = volue_data.shape[0]
    output_bit_depth = np.uint16
    images_in_sequence_n_bits = [None] * num_slices
    for slice in range(num_slices):
        slice_array = np.zeros(
            shape=(volue_data.shape[1], volue_data.shape[2]), dtype=output_bit_depth
        )
        slice_array[:, :] = np.multiply(
            volue_data[slice, :, :], np.iinfo(output_bit_depth).max
        ).astype(output_bit_depth)
        slice_data = np.transpose(slice_array)
        images_in_sequence_n_bits[slice] = slice_data

    image_out = tiff_2_png_converter.build_image_sequence(
        images_in_sequence_n_bits,
        volue_data.shape[1],
        volue_data.shape[2],
        num_slices,
        output_bit_depth,
    )
    pil_image_mode = "I;16"
    result_texture_atlas = Image.fromarray(image_out, mode=pil_image_mode)
    result_texture_atlas.save(save_file_path)
    return image_out.shape[1], image_out.shape[0], num_slices


def save_raw_data(raw_data: np.array, save_file_path: str):
    ## reshape to depth x height x width
    raw_data = raw_data.reshape(
        (raw_data.shape[2], raw_data.shape[1], raw_data.shape[0])
    )
    ## save osom-data data to file
    with open(save_file_path, "wb") as data_file:
        outData32 = raw_data.astype(np.float32)
        outData32.tofile(data_file)


def save_description_file(
    desc_file_path: str,
    data_width: int,
    data_height: int,
    data_depth: int,
    data_min_value: float,
    data_max_value: float,
    ocean_times,
):
    ## save description file
    with open(desc_file_path, "w") as desc_file:
        fprintf(
            desc_file,
            "%u,%u,%u,%.6f,%.6f\n",
            data_width,
            data_height,
            data_depth,
            data_min_value,
            data_max_value,
        )

        fprintf(desc_file, "%s\n", ocean_times.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    app()


if __name__ == "__main__":
    main()