import os
import tifffile as tiff
import numpy as np
from PIL import Image
import typer
import imageio
import logging

app = typer.Typer()


@app.command()
def raw_to_tiff(
    src: str = typer.Argument(..., help="Path to source raw file"),
    dest: str = typer.Argument(..., help="Path to Folder where tiffs will be saved"),
    width: int = typer.Argument(
        ..., help="Size of the second dimension of the binary 3D array"
    ),
    height: int = typer.Argument(
        ..., help="Size of the third dimension of the binary 3D array"
    ),
    depth: int = typer.Argument(
        ..., help="Size of first dimension of the binary 3D array"
    ),
    bit_depth: str = typer.Option(
        "8bit",
        help=" The data type and bits per pixel of the output image. 8bit (default) or 16bit",
    ),
):
    """
    This function converts a binary raw file that represents a 3D array to a sequence of tif images that depic a 3D volume

    Parameters:
                src (int): Path to binary raw file
                dest (int): Path to the folder of the resulting sequence of images
                width (int): Size of the second dimension of the binary 3D array
                height (int): Size of the third dimension of the binary 3D array
                depth (int): Size of first dimension of the binary 3D array
                bit_depth (int): Optional - Data type and number of bits per pixel of the output sequence of images 
                Options:
                    8bit (default)
                    16bit

    Returns:
        None. It writes a sequence of images in the dest path.

    """

    if not os.path.exists(src):
        raise ValueError(f"ERROR: Verify source path exists:\n{src}")
    if not os.path.exists(os.path.dirname(dest)):
        raise ValueError(
            f"ERROR: Verify destination path exists:\n{os.path.dirname(dest)}"
        )

    # By default we use 8-bit. Check if we need to change this setting
    output_bit_depth = np.uint8
    pil_image_mode = "L"
    if bit_depth == "16bit":
        output_bit_depth = np.uint16
        pil_image_mode = "I;16"

    # create a folder where the output will  be saved
    tiff_folder = os.path.join(dest, "tiff_data")
    if not (os.path.exists(tiff_folder)):
        os.mkdir(tiff_folder)

    typer.echo("Converting " + src + " to TIFF")
    with open(src, "rb") as raw_file:
        raw_data = np.fromfile(raw_file, dtype=np.float32)
        volume = raw_data.reshape((depth, height, width))
        digits = len(str(volume.shape[0] + 1))

        with typer.progressbar(
            range(volume.shape[0]), label="Processing raw "
        ) as m_depth:
            for slice in m_depth:
                slice_array = np.zeros(
                    shape=(volume.shape[1], volume.shape[2]), dtype=output_bit_depth
                )

                # Convert the raw bytes to the desire ouput format
                slice_array[:, :] = np.multiply(
                    volume[slice, :, :], np.iinfo(output_bit_depth).max
                ).astype(output_bit_depth)
                slice_image = np.transpose(slice_array)

                # save the sequence of images using tiff format
                raw_data_file_path, raw_data_filename = os.path.split(src)
                raw_data_filename, ext = os.path.splitext(raw_data_filename)
                tiff_filename = f"_{raw_data_filename}_slice{slice:0{digits}}"
                tiff_file = os.path.join(tiff_folder, tiff_filename + ".tif")

                im = Image.fromarray(slice_image, mode=pil_image_mode)
                im.save(tiff_file)

    typer.echo("Images written in " + dest)
    typer.echo("End")


def main():
    try:
        app()
    except ValueError:
        logging.error("Could not convert data to tiff.")
