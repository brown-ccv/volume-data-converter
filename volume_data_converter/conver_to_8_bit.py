import os
import cv2
from cv2 import log
import numpy as np
import imagesize
import math
import tifffile as tiff
from pyseq import Sequence

import typer

app = typer.Typer()


def bytes_scaling(img_array = np.array, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if img_array.dtype == np.uint8:
        return img_array

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = img_array.min()
    if cmax is None:
        cmax = img_array.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (img_array - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)



@app.command()
def scale_to_8_bit(
    src: str = typer.Argument(..., help="Path to file or folder to be processed"),
    dest: str = typer.Argument(..., help="Path to Folder where ouput will be saved")
):

    if not os.path.exists(dest):
         raise ValueError("ERROR: Verify destination path exists")
    
    if os.path.isfile(src):
         file_path, filename = os.path.split(dest)
         file_name,file_ext = os.path.splitext(filename)

         with open(src, "rb") as img_file:
            img_data = np.fromfile(img_file, dtype=np.float32)
            if img_data.dtype != np.uint8:
                # scale to 8 bit
                print("TODO")
    elif os.path.isdir(src):
        print("TODO")
    else:
        raise ValueError("ERROR: Verify src path exists")
   
    
