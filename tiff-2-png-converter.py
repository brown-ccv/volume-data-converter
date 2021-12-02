import argparse
import sys
import os 
import cv2
import numpy as np
import imagesize

def convert_tiff_directoryAction(path):
    list_files,width, height = list_folder_files(path)
    depth = len(list_files)
    image_r = np.zeros((width,height,depth))
    image_g = np.zeros((width,height,depth))
    image_b = np.zeros((width,height,depth))
    images = np.zeros((width,height,depth))
    for fname in list_files:
        if fname.endswith("ch1"):
             print("Load Image " + image_r.shape[2] + " Channel 1 - "+ fname )
        elif fname.endswith("ch2"):
             print("Load Image " + image_g.shape[2] + " Channel 2 - "+ fname )
        elif fname.endswith("ch3"):
             print("Load Image " + image_b.shape[2] + " Channel 3 - "+ fname )
        else:
             print("Load Image " + images.shape[2] + " RGB - "+ fname )
    #image_g
    #image_b
    #images
    


def list_folder_files(path_to_foder):

    file_list = []
    path = path_to_foder
    intitial_size_call = True
    width = -1
    height = -1
    if os.path.exists(path):
        for fname in os.listdir(path_to_foder):
            if fname.endswith(".tif") or fname.endswith(".tiff") or fname.endswith(".png"):
                file_full_path = os.path.join(path_to_foder, fname)
                n_width, n_height = imagesize.get(file_full_path)
                if intitial_size_call:
                  width, height = n_width,n_height
                  intitial_size_call = False
                elif width == n_width  and height == n_height:
                  file_list.append(file_full_path)
                else:
                    print("image " +file_full_path + "has a different size " + "("+str(width)+","+str(height)+")")
                    sys.exit()
    else:
        print("Folder does not exists")


    return file_list,width,height

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--dir",
                        required=True,
                        help="directory where the image sequence is located")
   args = parser.parse_args()
   if os.path.exists(args.dir):
       convert_tiff_directoryAction(args.dir)
   else:
       print('directory ' + args.dir +" does not exists")
       

if __name__ == '__main__':
    main()