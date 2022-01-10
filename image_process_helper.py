import numpy as np
import cv2

def gamma_correction(img: np.ndarray, gamma: float=1.0):
  img_dtype = img.dtype
  igamma = 1.0 / gamma

  imin, imax = img.min(), img.max()
  img_c = img.copy()
  img_c = ((img_c - imin) / (imax - imin)) ** igamma
  img_c = img_c * (imax - imin) + imin

  return img_c.astype(img_dtype)

def equalize_imgage_histogram(img):
    image_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_out_8bit = image_out.astype(np.uint8)
    equalizded_image = cv2.equalizeHist(image_out_8bit)
    return equalizded_image
    # img_dtype = img.dtype
    # flat_img = img.flatten()
    # hist,bins = np.histogram(img.flatten(),2**16,[0,2**16])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype(img_dtype)
    # return cdf