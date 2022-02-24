import argparse
import sys
import os
import cv2
from cv2 import log
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
import matplotlib.pyplot as plt

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
    image_out = np.zeros((new_resolution_image[0]*dim,new_resolution_image[1]*dim),dtype=img_type)

    with typer.progressbar(range(dim), label="Mapping slices to 2D") as progress:
        for i in progress:
            imgs_row = np.zeros((), dtype=img_type)
            for j in range(dim):
                tmp_img = np.zeros((new_resolution_image), dtype=img_type)
                
                if count < len(volume) and count <= num_slices:
                    tmp_img = cv2.resize( volume[count] , new_resolution_image)

                image_out[i*new_resolution_image[0]:(i*new_resolution_image[0])+new_resolution_image[0],
                          j*new_resolution_image[1]:(j*new_resolution_image[1])+new_resolution_image[1]] = tmp_img

                count = count + 1

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
    channel: int = typer.Option(
        0,
        help=" Channel of the data to export as single png",
    ),
    gamma: float = typer.Option(
        None,
        help=" Applies gamma correction to the dataset. Set a value > 0. Off by default",
    ),
    equilize_histogram: str = typer.Option(
        None, help=" Options: 'opencv' or 'custom' to select implementation"
    )
    
):
  

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

    list_files, width, height = list_folder_files(src)

    seqencer = Sequence(list_files)
    
    sequence_format = seqencer.format("%4l %r").split()
    ## parser.format returns a separated by space string describing the properties of the found sequence
    ## [0] sequence length
    ## [1] implied range, start-end  

    num_files_in_sequence = int(sequence_format[0])
    if num_files_in_sequence == 0:
        logging.error("No sequence found in source folder")
        raise ValueError("No sequence found")
    
    confirm_txt = "Found a sequence of "+str(num_files_in_sequence) + " images "+ sequence_format[1] + ". Do you want to proceed?"

    max_pixel = 0
    min_pixel = 0
    histogram = np.zeros((width * height *num_files_in_sequence ),dtype=np.uint16)
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
                    # flat = img.flatten()
                    # histogram[width * heigth * slice : width * heigth * (slice +1) ] = flat
                    max_pixel = (max_pixel, np.max(img))[np.max(img) > max_pixel]
                    min_pixel = (min_pixel, np.min(img))[np.min(img) < min_pixel]
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
                    if img.ndim > 2:
                        images_in_sequence[slice] = img[channel]
                    else:
                        images_in_sequence[slice] = img
        # plt.hist(histogram, bins=5) 
        # plt.title("Histogram with 'auto' bins")
        # plt.show()
        
        logging.info("MAX Value in volume: "+str(max_pixel))
        logging.info("MIN Value in volume: "+str(min_pixel))
        

        #convert to 8 bit
        images_in_sequence_8_bit = [None] * num_files_in_sequence
        with typer.progressbar(
            range(num_files_in_sequence), label="Rescaling to 8 bit signal"
            ) as progress:
                for slice in progress:
                    _img = images_in_sequence[slice]
                    img_8_bit = np.true_divide(_img,max_pixel)
                    images_in_sequence_8_bit[slice]= np.multiply(img_8_bit,255).astype(np.uint8)
            
        image_out = build_image_sequence(
                        images_in_sequence_8_bit, width, height,num_files_in_sequence ,np.uint8
                )
        
        
        name, extension = os.path.splitext(dest)
        if not extension:
                extension = ".png"
        
       
        # export as single channel image
        file_name = name + "_chn_8_bit" + str(channel) + extension
        logging.info(
                        "##Saving channel " + str(channel) + " Image to: " + file_name
                )
        cv2.imwrite(file_name, image_out)
        file = open(name + "_metadata", "a")
        file.write("Width:" + str(width) + "\n")
        file.write("Heigth:" + str(height) + "\n")
        file.write("Depth:" + str(num_files_in_sequence) + "\n")
        file.write("bps:" + str("uint8") + "\n")
        file.write("max:" + str(max_pixel) + "\n")
        file.write("max:" + str(min_pixel) + "\n")
        file.close()
        logging.info("Image written to file-system : " + name + extension)


if __name__ == "__main__":
    try:
        app()
    except ValueError:
        logging.error("Could not convert data to png.")
