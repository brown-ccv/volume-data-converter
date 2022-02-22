import argparse
import sys
import os
import cv2
import numpy as np
import imagesize
import math
import tifffile as tiff
import numpngw
import image_process_helper as iph
from scipy.interpolate import interp1d
import logging
import re
from collections.abc import Callable

from pyseq import Item, Sequence, diff, uncompress, get_sequences
from pyseq import SequenceError
import pyseq

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("tiff-2-png.log"), logging.StreamHandler()],
)

app = typer.Typer()



def list_folder_files(path_to_foder:str):

    file_list = []
    path = path_to_foder
    intitial_size_call = True
    width = -1
    height = -1
    if os.path.exists(path):
        for fname in os.listdir(path_to_foder):
            if (
                fname.endswith(".tif")
                or fname.endswith(".tiff")
                or fname.endswith(".png")
            ):
                file_full_path = os.path.join(path_to_foder, fname)
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


def interpolate_data(img_array: np.array, width: int, heigth: int, num_images: int):
    ## STILL IN DEVELOP - NOT SURE IF WE NEED IT
    # ptython list to numpy 3d array
    img_3d_array = np.array((width, heigth))
    for z in range(num_images):
        img_2d_array = img_array[z]
        if z == 0:
            img_3d_array = img_2d_array
        elif img_2d_array is not None:
            img_3d_array = np.dstack([img_3d_array, img_2d_array])
        else:
            img_3d_array = np.dstack([img_3d_array, np.zeros((width, heigth))])
    min_data = np.amin(np.amin(img_3d_array, axis=2))
    max_data = np.amax(np.amax(img_3d_array, axis=2))
    logging.info("Minimum value in the whole array:%d" % (min_data))
    logging.info("Maximum value in the whole array:%d" % (max_data))
    min_max_scale = np.arange(min_data, max_data)


#this function will be removed 
def merge_rgb(
    image_channels: np.array,
    rgb_images: list,
    width: int,
    height: int,
    z_slices: int,
    bits_per_sample: np.dtype,
    equalize_histogram:np.array
):
    zero_image = np.zeros((width, height), dtype=bits_per_sample)
    with typer.progressbar(
        range(z_slices), label="Mergin channels"
    ) as progress:
        for z in progress:
            r_channel = np.zeros((width, height), dtype=bits_per_sample)
            g_channel = np.zeros((width, height), dtype=bits_per_sample)
            b_channel = np.zeros((width, height), dtype=bits_per_sample)
            current_image_channels = [r_channel, g_channel, b_channel]
            ## loop over the channels :  r = 0, g = 1, b =2
            for i in range(3):
                current_channel = image_channels[i]
                if current_channel[z] is not None:
                    current_image_channels[i] = current_channel[z]
                    if equalize_histogram is not None:
                        current_image_channels[i] = equalize_histogram(
                            current_image_channels[i]
                        )
                else:
                    current_image_channels[i] = zero_image
            rgb_images[z] = cv2.merge(current_image_channels)

     
             


def build_image_sequence(
    volume: np.array, width: int, height: int, z_slices: int, img_type: np.dtype
):
    num_slices = z_slices + 1
    dim = math.ceil(math.sqrt(num_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil(max(width, height) / max_res)

    logging.info("resolution_image_before " + str(width) + " " + str(height))
    logging.info("downscale " + str(downscale))

    new_resolution_image = (int(width / downscale), int(height / downscale))
    count = 0
    image_out = np.zeros((dim*new_resolution_image))

    with typer.progressbar(range(dim), label="Mapping slices to 2D") as progress:
        for i in progress:
            imgs_row = np.zeros((), dtype=img_type)
            for j in range(dim):
                tmp_img = np.zeros((new_resolution_image), dtype=img_type)
                img_index = (0, count)[
                    count < len(volume) and count <= z_slices
                ]
                img = volume[img_index]

                if img is not None:
                    tmp_img = cv2.resize(img, new_resolution_image)

                if img_index == 0:
                    tmp_img[:, :] = 0

                if j == 0:
                    imgs_row = np.copy(tmp_img)
                else:
                    imgs_row = cv2.hconcat([imgs_row, tmp_img])
                count = count + 1

            if i == 0:
                image_out = np.copy(imgs_row)
            else:
                image_out = cv2.vconcat([image_out, imgs_row])

    return image_out


def save_rgb_png_image(
    volume: np.array,
    width: int,
    height: int,
    dest: str,
    z_slices: int,
    img_type: np.dtype,
):
    image_out = build_image_sequence(volume, width, height, img_type)
    name, extension = os.path.splitext(dest)
    if not extension:
        extension = ".png"

    amax = np.amax(image_out)
    norm = np.zeros((image_out.shape))
    normalized_image_u16bit = cv2.normalize(
        image_out, norm, 0, 2 ** 16, cv2.NORM_MINMAX
    )
    numpngw.write_png(name + extension, image_out)
    numpngw.write_png(name + "_normalized" + extension, normalized_image_u16bit)

    file = open(name + "_metadata", "a")
    file.write("Width:" + str(width) + "\n")
    file.write("Heigth:" + str(height) + "\n")
    file.write("Depth:" + str(z_slices) + "\n")
    file.write("bps:" + str(image_out.dtype) + "\n")
    file.write("amax:" + str(amax) + "\n")
    file.close()
    logging.info("Image written to file-system : " + name + extension)


@app.command()
def convert_to_png(
    src: str = typer.Argument(..., help="Path to source folder"),
    dest: str = typer.Argument(..., help="Path where png will be saved"),
    slices: int = typer.Option(
        None,
        help=" Number of images in your stack. It will take all the files in the src folder by default",
    ),
    verbose: bool = typer.Option(
        False,
        help=" Output log messages on console. By default it will print to .log file",
    ),
    channel: int = typer.Option(
        False,
        help=" Channel of the data to export as single png",
    ),
    gamma: float = typer.Option(
        None,
        help=" Applies gamma correction to the dataset. Set a value > 0. Off by default",
    ),
    equilize_histogram: str = typer.Option(
        None, help=" Options: 'opencv' or 'custom' to select implementation"
    ),
):
  
    # parser = Seqparse()
    # parser.scan_path(src)
    # for item in parser.output():
    #  print(str(item))

    name, extension = os.path.splitext(dest)
    if not (os.path.exists(src) and os.path.exists(os.path.dirname(name))):
        raise ValueError("ERROR: Verify source and destination paths exists")

    equilize_histogram_function = None

    if equilize_histogram is not None:
        if equilize_histogram == "opencv":
           equilize_histogram_function = iph.equalize_image_histogram_opencv
        elif equilize_histogram == "custom":
            equilize_histogram_function = iph.equalize_image_histogram_custom
        else:
            raise ValueError("ERROR: equilize_histogram value is not valid")

    list_files, width, heigth = list_folder_files(src)

    seqencer = Sequence(list_files)
    
    sequence_format = seqencer.format("%4l %r")
    ## parser.format returns a separated by space string describing the properties of the found sequence
    ## [0] sequence length
    ## [1] implied range, start-end  

    num_files_in_sequence = sequence_format[0]
    if num_files_in_sequence == 0:
        logging.error("No sequence found in source folder")
        raise ValueError("No sequence found")
    
    confirm_txt = "Found a sequence of "+str(num_files_in_sequence) + " images "+ sequence_format[2] + ". Do you want to proceed?"

    
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
                    elif bits_per_sample != img.dtype:
                        raise ValueError(
                        "image "
                        + file_full_path
                        + "has a different sample type "
                        + bits_per_sample
                        + " expected: "
                        + bits_per_sample
                        )
                    images_in_sequence[slice] = img[channel]
                
    name, extension = os.path.splitext(dest)
    if not extension:
            extension = ".png"
    
    # saving to 8 bit
    
    file_name = name + "_chn_" + str(channel + 1) + extension
    image_out = build_image_sequence(
                    images_in_sequence, width, heigth, bits_per_sample
            )
    logging.info(
                    "##Saving channel " + str(channel) + " Image to: " + file_name
            )
    cv2.imwrite(file_name, image_out)


if __name__ == "__main__":
    try:
        app()
    except ValueError:
        logging.error("Could not convert data to png.")
