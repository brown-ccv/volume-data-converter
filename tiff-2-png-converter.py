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

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("tiff-2-png.log"), logging.StreamHandler()],
)

app = typer.Typer()

arg_values = {
    "verbose": False,
    "equalize_histogram": False,
    "equalize_histogram_method": None,
    "gamma": False,
    "gamma_val": 1,
    "nimgs": None,
    "source": "",
    "dest": "",
}


def list_folder_files(path_to_foder):

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


def interpolate_data(img_array, width, heigth, num_images):
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


def merge_rgb(image_channels, rgb_images, width, height, bits_per_sample):
    zero_image = np.zeros((width, height), dtype=bits_per_sample)
    with typer.progressbar(
        range(arg_values["nimgs"]), label="Mergin channels"
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
                    if arg_values["equalize_histogram"]:
                        current_image_channels[i] = arg_values[
                            "equalize_histogram_method"
                        ](current_image_channels[i])
                else:
                    current_image_channels[i] = zero_image

            merged_image = cv2.merge(current_image_channels)

            if arg_values["gamma"] is not None:
                float_gamma = float(arg_values["gamma"])
                merged_image = iph.gamma_correction(merged_image, float_gamma)
            rgb_images[z] = merged_image


def build_image_sequence(volume, width, height):
    z_slices = arg_values["nimgs"] + 1
    dim = math.ceil(math.sqrt(z_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil(max(width, height) / max_res)
    if arg_values["verbose"]:
        logging.info("resolution_image_before " + str(width) + " " + str(height))
        logging.info("downscale " + str(downscale))

    resolution_image = (int(width / downscale), int(height / downscale))
    count = 0
    image_out = np.zeros(())
    
    with typer.progressbar(range(dim), label="Mapping slices to 2D") as progress:
        for i in progress:
            imgs_row = np.zeros(())
            for j in range(dim):
                tmp_img = np.zeros((resolution_image))
                if count < len(volume) and count <= arg_values["nimgs"]:
                    img = volume[count]
                    if img is None:
                        img = zero_image = np.zeros((width, height), dtype=uint8)
                    tmp_img = cv2.resize(img, resolution_image)
                else:
                    img = volume[0]
                    tmp_img = cv2.resize(img, resolution_image)
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


def save_rgb_png_image(volume, width, height):
    image_out = build_image_sequence(volume, width, height)
    name, extension = os.path.splitext(arg_values["dest"])
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
    file.write("Depth:" + str(arg_values["nimgs"]) + "\n")
    file.write("bps:" + str(image_out.dtype) + "\n")
    file.write("amax:" + str(amax) + "\n")
    file.close()
    logging.info("Image written to file-system : " + name + extension)


@app.command()
def convert_to_png(
    src: str = typer.Argument(..., help="Path to source folder"),
    dest: str = typer.Argument(..., help="Path where png will be saved"),
    nimgs: int = typer.Option(
        None,
        help=" Number of images in your stack. It will take all the files in the src folder by default",
    ),
    verbose: bool = typer.Option(
        False,
        help=" Output log messages on console. By default it will print to .log file",
    ),
    merge: bool = typer.Option(
        False,
        help=" Merge the channels into a single rgb image. False by default, creating an image png per channel",
    ),
    gamma: float = typer.Option(
        None,
        help=" Applies gamma correction to the dataset. Set a value > 0. Off by default",
    ),
    equilize_histogram: str = typer.Option(
        None, help=" Options: 'opencv' or 'custom' to select implementation"
    ),
):
    arg_values["src"] = src
    arg_values["dest"] = dest
    arg_values["nimgs"] = nimgs
    arg_values["verbose"] = verbose
    arg_values["merge"] = merge
    arg_values["gamma"] = gamma
    arg_values["equilize_histogram"] = equilize_histogram

    name, extension = os.path.splitext(dest)
    if not (os.path.exists(src) and os.path.exists(os.path.dirname(name))):
        raise ValueError("ERROR: Verify source and destination paths exists")

    if equilize_histogram is not None:
        if equilize_histogram == "opencv":
            arg_values[
                "equalize_histogram_method"
            ] = iph.equalize_image_histogram_opencv
        elif equilize_histogram == "custom":
            arg_values[
                "equalize_histogram_method"
            ] = iph.equalize_image_histogram_custom
        else:
            raise ValueError("ERROR: equilize_histogram value is not valid")

    list_files, width, heigth = list_folder_files(src)
    if nimgs is None:
        nimgs = len(list_files)
    image_r = [None] * nimgs
    image_g = [None] * nimgs
    image_b = [None] * nimgs
    images = [None] * nimgs
    bits_per_sample = ""
    with typer.progressbar(
        range(len(list_files)), label="grouping channels"
    ) as progress:
        for i in progress:
            file_full_path = list_files[i]
            img = tiff.imread(list_files[i])
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

            name, extension = os.path.splitext(file_full_path)
            file_name = os.path.basename(name)
            sequence_number = re.search("^p[0-9]{1,5}", file_name)
            sequence_number_int = int(sequence_number.group(0)[1:])
            if "ch1" in name:
                if verbose:
                    logging.info(
                        "Load Image "
                        + str(sequence_number_int)
                        + " Channel 1 - "
                        + file_full_path
                    )
                image_r[sequence_number_int - 1] = img
            elif "ch2" in name:
                if verbose:
                    logging.info(
                        "Load Image "
                        + str(sequence_number_int)
                        + " Channel 2 - "
                        + file_full_path
                    )
                image_g[sequence_number_int - 1] = img
            elif "ch3" in name:
                if verbose:
                    logging.info(
                        "Load Image "
                        + str(sequence_number_int)
                        + " Channel 3 - "
                        + file_full_path
                    )
                image_b[sequence_number_int - 1] = img
            else:
                if verbose:
                    logging.info(
                        "Load Image "
                        + str(sequence_number_int)
                        + " RGB - "
                        + file_full_path
                    )
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[sequence_number_int - 1] = image_rgb
    
    image_channels = [image_r, image_g, image_b]
    if merge:
        if len(image_r) != 0 or len(image_g) != 0 or len(image_b) != 0:        
            merge_rgb(image_channels, images, width, heigth, bits_per_sample)
            logging.info("##Saving Merged Image to: " + dest)
            save_rgb_png_image(images, width, heigth)
    else:
        name, extension = os.path.splitext(arg_values["dest"])
        if not extension:
            extension = ".png"
        # saving to 8 bit
        for channel in range(3):
            if len(image_channels[channel]) != 0:
                image_out = build_image_sequence(image_channels[channel], width, heigth)
                cv2.imwrite("chn_"+ str(channel+1)+"_"+ name + extension, image_out[:, :, 1])


if __name__ == "__main__":
    try:
        app()
    except ValueError:
        logging.error("Could not convert data to png.")
