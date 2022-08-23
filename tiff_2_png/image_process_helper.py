import numpy as np
import cv2


def gamma_correction(img: np.ndarray, gamma: float = 1.0):
    """
    Custom Implementation of gamma correction to process luminance on images

    """
    img_dtype = img.dtype
    igamma = 1.0 / gamma

    imin, imax = img.min(), img.max()
    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** igamma
    img_c = img_c * (imax - imin) + imin

    return img_c.astype(img_dtype)


def equalize_image_histogram_opencv(img_array):
    """
    Use opencv implementation of the equalize histogram

    The official documentation of opencv equalizeHist:
    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

    """
    img_dtype = img_array.dtype
    image_out_8bit = img_array.astype(np.uint8)
    equalizded_image = cv2.equalizeHist(image_out_8bit)
    return equalizded_image.astype(img_dtype)


def equalize_image_histogram_custom(img_array):
    """
    Custom implementation for redistributing intensity values over the histogram, increasing the global contrast of the image.

    Followed this code:
    https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43

    Experimentally, this method brights up the image more that the opencv implementation.
    """
    img_dtype = img_array.dtype
    histogram_array = np.bincount(
        img_array.flatten(), minlength=np.iinfo(img_dtype).max
    )
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(np.iinfo(img_dtype).max * chistogram_array).astype(
        np.uint8
    )
    img_list = list(img_array.flatten())
    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    return eq_img_array.astype(img_dtype)
