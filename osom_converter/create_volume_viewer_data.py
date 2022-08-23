from netCDF4 import Dataset
import typer
import numpy as np
import os
from datetime import datetime, timedelta
from osom_converter import scoord_du_new2 as scoor_du

app = typer.Typer()


def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)


@app.command()
def createOsomData(
    osom_gridfile: str, osom_data_file: str, output_folder: str, data_descriptor: str
):

    """
    Converts NETCDF files (*.nc) form the osom model to data files that can be read
    in the volume viewer desktop app

    osom_gridfile: grid file providing uv transformation coordinates ( i.e: file provided by this tool osom_grid4_mindep_smlp_mod7.nc)
    osom_data_file: osom data file with multi-variable data
    output_folder: location where the resulting data will be saved
    data_descriptor: variable to extract from the osom data file (temp, salt)

    """
    # Default factors/scalers
    verticalLevels = 15
    downscaleFactor = 2

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
    data = nc_dataFile.variables[data_descriptor][:]
    zeta = nc_dataFile.variables["zeta"][:]
    vtransform = nc_dataFile.variables["Vtransform"][:]
    vstretching = nc_dataFile.variables["Vstretching"][:]
    theta_s = nc_dataFile.variables["theta_s"][:]
    theta_b = nc_dataFile.variables["theta_b"][:]
    hc = nc_dataFile.variables["hc"][:]
    ocean_time = nc_dataFile.variables["ocean_time"][:]

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

    output_f = os.path.join(output_folder, "data")
    if not os.path.exists(output_f):
        os.mkdir(output_f)

    start_time = datetime.strptime("2006-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    typer.echo(" Converting nc data to raw and desc files. This process will take time")
    with typer.progressbar(
        range(np.shape(data)[0]), label="Processing Time"
    ) as time_progress:
        for time in time_progress:
            # intiialize with zero
            out_data = np.zeros((slices, np.shape(x_rho)[0], np.shape(x_rho)[1]))
            progress_bar_label = "Processing Time Step " + str(time + 1)
            with typer.progressbar(
                range(np.shape(x_rho)[0]), label=progress_bar_label
            ) as gridx_progress:
                for x in gridx_progress:
                    for y in range(np.shape(x_rho)[1]):
                        # get depth column
                        t_data = data[time, :, x, y]
                        t_data = np.ma.squeeze(t_data)
                        idx = ~t_data.mask
                        if np.sum(idx.astype(int)) > 0:
                            # read set depth
                            domain = z[idx, x, y]
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
                f"{data_descriptor}_{osom_data_filename}_timestep{time+1:0{digits}}"
            )

            with open(
                os.path.join(output_f, data_filename + ".raw"), "wb"
            ) as data_file:
                outData32 = out_data.astype(np.float32)
                outData32.tofile(data_file)

            ## save description file
            with open(
                os.path.join(output_f, data_filename + ".desc"), "w"
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
                current_time = start_time + timedelta(0, ocean_time[time])
                fprintf(desc_file, "%i\n", datetime.timestamp(current_time))
                fprintf(desc_file, "%s\n", current_time.strftime("%m/%d/%Y-%H:%M:%S"))


def main():
    app()
