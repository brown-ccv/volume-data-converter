import os
import cv2
from cv2 import log
import numpy as np
import imagesize
import math
import tifffile as tiff
from tiff_2_png import image_process_helper as iph
import logging
import matplotlib.pyplot as plt
import mplhep as hep

from pyseq import Sequence

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("tiff-2-png.log"), logging.StreamHandler()],
)

app = typer.Typer()


def list_folder_files(path_to_folder: str):

    """
    Checks for tiff files in a folder and returns a python list of the full paths.
    Parameters:
                path_to_foder (str): python list  of 2d images

    Returns:
        Python list of string with the full paths to the images in the folder

    """

    file_list = []
    path = path_to_folder
    intitial_size_call = True
    width = -1
    height = -1
    if os.path.exists(path):
        for fname in os.listdir(path_to_folder):
            if fname.endswith(".tif") or fname.endswith(".tiff"):
                file_full_path = os.path.join(path_to_folder, fname)
                n_width, n_height = imagesize.get(file_full_path)
                if intitial_size_call:
                    width, height = n_width, n_height
                    intitial_size_call = False
                    file_list.append(file_full_path)
                elif width == n_width and height == n_height:
                    file_list.append(file_full_path)
                else:
                    raise (
                        "image "
                        + file_full_path
                        + "has a different size "
                        + "("
                        + str(width)
                        + ","
                        + str(height)
                        + ")"
                    )

    else:
        logging.error("Folder " + path + " does not exists")

    return file_list, height, width


def build_image_sequence(
    volume: np.array, width: int, height: int, z_slices: int, img_type: np.dtype
):
    """
    Converts a python list of images (2d arrays) into a 2D texture map
    Parameters:
                volume (np.array): python list  of 2d images
                width (int): images width
        height (int): images height
        z_slices (int): number of images in the stack
        img_type (np.dtype): bits per sample in the images
    Returns:
        2D np.array. Tiff images sequentially placed in a texture map

    """
    num_slices = z_slices + 1
    dim = math.ceil(math.sqrt(num_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil(max(width, height) / max_res)

    logging.info("resolution_image_before " + str(width) + " " + str(height))
    logging.info("downscale " + str(downscale))

    new_resolution_image = (int(width / downscale), int(height / downscale))
    count = 0
    image_out = np.zeros(
        (new_resolution_image[0] * dim, new_resolution_image[1] * dim), dtype=img_type
    )

    with typer.progressbar(range(dim), label="Mapping slices to 2D") as progress:
        for i in progress:
            imgs_row = np.zeros((), dtype=img_type)
            for j in range(dim):
                tmp_img = np.zeros((new_resolution_image), dtype=img_type)

                if count < len(volume) and count <= num_slices:
                    tmp_img = cv2.resize(volume[count], new_resolution_image)

                image_out[
                    i * new_resolution_image[0] : (i * new_resolution_image[0])
                    + new_resolution_image[0],
                    j * new_resolution_image[1] : (j * new_resolution_image[1])
                    + new_resolution_image[1],
                ] = tmp_img

                count = count + 1

    return image_out


@app.command()
def convert_to_png(
    src: str = typer.Argument(..., help="Path to source folder"),
    dest: str = typer.Argument(..., help="Path where png will be saved"),
    channel: int = typer.Option(
        0,
        help=" Channel of the data to export as single png",
    ),
    gamma: float = typer.Option(
        None,
        help=" Applies gamma correction to the dataset. Set a value > 0. Off by default",
    ),
    equalize_histogram: str = typer.Option(
        None, help=" Options: 'opencv' or 'custom' to select implementation"
    ),
    histogram: bool = typer.Option(
        False,
        help=" Write the histogram of the resulting image in a separate text file",
    ),
):

    """
    This function converts a tiff stack of images (volume) to a 2D texture map to be
    loaded in the volume viewer.

    Parameters:
                src (int): Path to tiff stack root folder
                dest (int): Path + filename of the resulting 2D texture map.
        channel (int): On multi channel of the images (i.e rgb), the specific
                       channel to read from. If the image only has 1 channel it reads from channel 0.

    Returns:
        None. It writes an image in the dest path. An additional meta-file is written with the information of the dataset.

    """
    dest_name, extension = os.path.splitext(dest)
    if not (os.path.exists(src) and os.path.exists(os.path.dirname(dest_name))):
        raise ValueError("ERROR: Verify source and destination paths exists")

    equilize_histogram_function = None

    if equalize_histogram is not None:
        if equalize_histogram == "opencv":
            equilize_histogram_function = iph.equalize_image_histogram_opencv
        elif equalize_histogram == "custom":
            equilize_histogram_function = iph.equalize_image_histogram_custom
        else:
            raise ValueError("ERROR: equilize_histogram value is not valid")

    list_files, width, height = list_folder_files(src)

    seqencer = Sequence(list_files)

    sequence_format = seqencer.format("%4l %r %m %R").split()
    ## parser.format returns a separated by space string describing the properties of the found sequence
    ## [0] sequence length
    ## [1] implied range, start-end

    num_files_in_sequence = int(sequence_format[0])
    if num_files_in_sequence == 0:
        logging.error("No sequence found in source folder")
        raise ValueError("No sequence found")

    confirm_txt = (
        "Found a sequence of "
        + str(num_files_in_sequence)
        + " images "
        + sequence_format[1]
        + ". Do you want to proceed?"
    )

    global_max_pixel = 0
    global_min_pixel = 0
    # histogram = np.zeros((width * height * num_files_in_sequence), dtype=np.uint16)
    if typer.confirm(confirm_txt, abort=False):
        images_in_sequence = [None] * num_files_in_sequence
        bits_per_sample = ""
        with typer.progressbar(
            range(num_files_in_sequence), label="Reading images from directory"
        ) as progress:
            for slice in progress:
                file_full_path = list_files[slice]
                if seqencer.contains(file_full_path):
                    img = tiff.imread(list_files[slice])

                    ## check all images have the same data type
                    if bits_per_sample == "":
                        bits_per_sample = img.dtype
                        global_max_pixel = np.iinfo(img.dtype).min
                        global_min_pixel = np.iinfo(img.dtype).max
                    elif bits_per_sample != img.dtype:
                        raise ValueError(
                            "image "
                            + file_full_path
                            + "has a different sample type "
                            + bits_per_sample
                            + " expected: "
                            + bits_per_sample
                        )
                    if img.ndim > 2:
                        images_in_sequence[slice] = img[channel]
                    else:
                        images_in_sequence[slice] = img
                    global_max_pixel = (global_max_pixel, np.max(img))[
                        np.max(img) > global_max_pixel
                    ]
                    global_min_pixel = (global_min_pixel, np.min(img))[
                        np.min(img) < global_min_pixel
                    ]

        logging.info(f"MAX Value in volume: {global_max_pixel}")
        logging.info(f"MIN Value in volume: {global_min_pixel}")

        # convert to 8 bit
        images_in_sequence_8_bit = [None] * num_files_in_sequence
        with typer.progressbar(
            range(num_files_in_sequence), label="Rescaling to 8 bit signal"
        ) as progress:
            for slice in progress:
                _img = images_in_sequence[slice]
                img_8_bit = np.subtract(_img, global_min_pixel)
                img_8_bit = np.true_divide(_img, global_max_pixel - global_min_pixel)
                img_8_bit = np.multiply(img_8_bit, 255).astype(np.uint8)
                if gamma is not None:
                    img_8_bit = iph.gamma_correction(img_8_bit, gamma)
                images_in_sequence_8_bit[slice] = img_8_bit

        image_out = build_image_sequence(
            images_in_sequence_8_bit, width, height, num_files_in_sequence, np.uint8
        )

        ## split path to resolve path of multiple files to be written
        parent_folder = os.path.abspath(os.path.join(dest, os.pardir))
        file_base_name = os.path.basename(dest)
        file_name, extension = os.path.splitext(file_base_name)
        if not extension:
            extension = ".png"

        # export as single channel image
        file_name = file_name + "_chn_" + str(channel)
        file_name_full_path = os.path.join(parent_folder, file_name + extension)
        logging.info(
            "##Saving channel " + str(channel) + " Image to: " + file_name_full_path
        )
        cv2.imwrite(file_name_full_path, image_out)
        # write metadata file
        metadata_file_path = os.path.join(parent_folder, file_name + "_metadata")
        file = open(metadata_file_path, "a")
        file.write("Width:" + str(width) + "\n")
        file.write("Heigth:" + str(height) + "\n")
        file.write("Depth:" + str(num_files_in_sequence) + "\n")
        file.write("bps:" + str("uint8") + "\n")
        file.write("max:" + str(global_max_pixel) + "\n")
        file.write("max:" + str(global_min_pixel) + "\n")
        file.close()
        logging.info("Image written to file-system : " + file_base_name)

        if histogram:
            img_histogram = np.histogram(image_out, bins=255)
            histogram_file_full_path = os.path.join(
                parent_folder, file_name + "_Histogram.txt"
            )
            np.savetxt(histogram_file_full_path, img_histogram[0], fmt="%u")
            logging.info(
                "Histogram written to file-system : " + histogram_file_full_path
            )
            hep.histplot(H=img_histogram[0], bins=img_histogram[1])
            # Wait for all figures to be closed before returning.
            plt.show(block=True)


def main():
    try:
        app()
    except ValueError:
        logging.error("Could not convert data to png.")
