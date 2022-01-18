
import argparse
import sys
import os 
import cv2
import numpy as np
import imagesize
import math
import tifffile  as tiff
import numpngw
from progress_bar import *
from image_process_helper import *
from scipy.interpolate import interp1d
import re

verbose = False
equalize_histogram = False
equalize_histogram_method = None
gamma = False
gamma_val = 1


def convert_tiff_directory_action(source_path, dest_path, num_images=None):
    list_files,width, heigth = list_folder_files(source_path)
    if num_images is None:
        num_images = len(list_files)
    image_r = [None] * num_images
    image_g = [None] * num_images
    image_b = [None] * num_images
    images = [None] * num_images
    bits_per_sample = ""
    printProgressBar(0, num_images, prefix = 'grouping channels:', suffix = 'Complete', length = 50)
    for i in range(len(list_files)):
        file_full_path = list_files[i]
        img = tiff.imread(list_files[i])
        ## check all images have the same data type
        if bits_per_sample=="":
            bits_per_sample = img.dtype
        elif bits_per_sample!=img.dtype:
            print("image " + file_full_path + "has a different sample type "+bits_per_sample+" expected: "+bits_per_sample)
            sys.exit("Check images bits per sample tag")
        
        name, extension = os.path.splitext(file_full_path)
        file_name  = os.path.basename(name)
        sequence_number = re.search("^p[0-9]{1,5}",file_name)
        sequence_number_int = int(sequence_number.group(0)[1:])
        print(sequence_number_int )
        if "ch1" in name:
             if verbose:
                print("Load Image " + str(sequence_number_int) + " Channel 1 - "+ file_full_path )
             image_r[sequence_number_int - 1] = img
        elif "ch2" in name:
             if verbose:
                print("Load Image " + str(sequence_number_int) + " Channel 2 - "+ file_full_path )
             image_g[sequence_number_int - 1] = img
        elif "ch3" in name:
             if verbose:
                print("Load Image " + str(sequence_number_int) + " Channel 3 - "+ file_full_path )
             image_b[sequence_number_int - 1] = img
        else:
             if verbose:
                print("Load Image " + str(sequence_number_int) + " RGB - "+ file_full_path )
             image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
             images[sequence_number_int - 1] = image_rgb
        printProgressBar(i + 1, num_images, prefix = 'grouping channels:', suffix = 'Complete', length = 50)    

    #interpolate
    interpolateData(image_g,width,heigth,num_images)

    if len(image_r) != 0 or len(image_g) != 0 or len(image_b) != 0:
        merge_RGB(image_r, image_g, image_b, images,width,heigth,num_images,bits_per_sample)

    print("##Saving Image to: "+dest_path)
    save_to_Image(images,num_images,width, heigth,dest_path)

def interpolateData(img_array, width,heigth, num_images):
    #ptython list to numpy 3d array
    img_3d_array = np.array((width,heigth))
    for z in range(num_images):
        img_2d_array = img_array[z]
        if z == 0:
            img_3d_array = img_2d_array
        elif img_2d_array is not None:
            img_3d_array = np.dstack([img_3d_array,img_2d_array])
        else:
            img_3d_array = np.dstack([img_3d_array,np.zeros((width,heigth))])
    min_data = np.amin(np.amin(img_3d_array,axis=2))
    max_data = np.amax(np.amax(img_3d_array,axis=2))
    print("Minimum value in the whole array:%d"%(min_data))   
    print("Maximum value in the whole array:%d"%(max_data)) 
    min_max_scale = np.arange(min_data, max_data)
    
    #reshaped_array = np_array.reshape(4, 2)
    #print("")

def save_to_Image(volume,depth,width,height, dest_path):

    z_slices = depth + 1
    dim = math.ceil(math.sqrt(z_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil( max(width, height) / max_res)
    if verbose:
        print("resolution_image_before " + str(width)+ " "+str( height))  
        print("downscale " + str(downscale))  
	
    resolution_image = ( int(width / downscale), int(height / downscale))
    
    count = 0
    image_out = np.zeros(())
    printProgressBar(0, dim, prefix = 'Mapping slices to 2D:', suffix = 'Complete', length = 50)
    for i in range(dim):
        imgs_row = np.zeros(())
        for j in range(dim):
            tmp_img = np.zeros((resolution_image))
            if count < len(volume) and count <= depth:
               img =volume[count]
               tmp_img =  cv2.resize(img, resolution_image)
            else:          
                img =volume[0]     
                tmp_img =  cv2.resize(img,resolution_image)
                tmp_img[:,:] = 0
            if j == 0:
                imgs_row = np.copy(tmp_img)
            else:
                imgs_row = cv2.hconcat([imgs_row,tmp_img])
            count = count+1
        if i == 0:
            image_out = np.copy(imgs_row)
        else:
            image_out = cv2.vconcat([image_out,imgs_row])
    printProgressBar(i+1, dim, prefix = 'Mapping slices to 2D:', suffix = 'Complete', length = 50)
    
    name, extension = os.path.splitext(dest_path)
    if not extension:
        extension = ".png"
    
    amax = np.amax(image_out)    
    norm = np.zeros((image_out.shape))
    normalized_image_u16bit = cv2.normalize(image_out,norm,0,2**16,cv2.NORM_MINMAX)
    numpngw.write_png(name+extension,image_out)
    numpngw.write_png(name+"_normalized"+extension,normalized_image_u16bit)
    
    file = open(name+"_metadata", "a")
    file.write("Width:"+str(width)+"\n")
    file.write("Heigth:"+str(height)+"\n")
    file.write("Depth:"+str(depth)+"\n")
    file.write("bps:"+str(image_out.dtype)+"\n")
    file.write("amax:"+str(amax)+"\n")
    #f.write("amax normalized:"+str(amax2)+"\n")
    file.close()
    print("Image written to file-system : "+name+extension)
    
    
def merge_RGB(image_r,  image_g,  image_b, rgb_images, width, heigh, depth,bits_per_sample):
    zero_image = np.zeros((width,heigh),dtype=bits_per_sample)
    printProgressBar(0, depth, prefix = 'Mergin channels:', suffix = 'Complete', length = 50)
    for z in range(depth):
        r_channel = np.zeros((width,heigh),dtype=bits_per_sample)
        g_channel = np.zeros((width,heigh),dtype=bits_per_sample)
        b_channel = np.zeros((width,heigh),dtype=bits_per_sample)
        if image_r[z] is not None:
            r_channel = image_r[z]
            if equalize_histogram:
                r_channel = equalize_histogram_method(r_channel)
        else:
            r_channel = zero_image
        if image_g[z] is not None:
            g_channel = image_g[z]
            if equalize_histogram:
                g_channel = equalize_histogram_method(g_channel)
        else:
            g_channel = zero_image
        if image_b[z] is not None:
            b_channel = image_b[z]
            if equalize_histogram:
                b_channel = equalize_histogram_method(b_channel)
        else:
            b_channel = zero_image
        

        merged_image = cv2.merge([r_channel,g_channel,b_channel])
        
        if gamma:
            merged_image = gamma_correction(merged_image,gamma_val)
        rgb_images[z]=  merged_image



        printProgressBar(z+1, depth, prefix = 'Mergin channels:', suffix = 'Complete', length = 50)
    
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
                  file_list.append(file_full_path)
                elif width == n_width  and height == n_height:
                  file_list.append(file_full_path)
                else:
                    print("image " +file_full_path + "has a different size " + "("+str(width)+","+str(height)+")")
                    sys.exit()
    else:
        print("Folder does not exists")


    return file_list,height,width

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("-s", "--source",
                        required=True,
                        help="Path to directory where the image sequence is located")
   parser.add_argument("-d", "--dest",
                        required=True,
                        help="Path to the directory and filename of the result merged image (png by default) will be saved. ")
   parser.add_argument("--verbose",
                        help="Generate messages ouput ",action='store_true')
   
   parser.add_argument("--num_imgs",
                        help="Generate messages ouput ")

   parser.add_argument("--equalize1",
                        help="equilize histogram in volume slices using opencv functionality",action='store_true')
   
   parser.add_argument("--equalize2",
                        help="equilize histogram in volume slices using custom implementation",action='store_true')
   
   parser.add_argument("--gamma",
                        help="Apply gamma correction")
                        
   args = parser.parse_args()
   name, extension = os.path.splitext(args.dest)

   if args.verbose:
       global verbose
       verbose = True

   if args.equalize1 or args.equalize2:
       global equalize_histogram
       global equalize_histogram_method
       equalize_histogram = True
       if args.equalize1:
           equalize_histogram_method = equalize_imgage_histogram_opencv
       if args.equalize2:
           equalize_histogram_method = equalize_imgage_histogram_custom

   num_imgs = None
   if args.num_imgs:
       num_imgs = int(args.num_imgs)
    
   if args.gamma:
       global gamma,gamma_val
       gamma = True
       gamma_val = args.gamma

   if os.path.exists(args.source) and os.path.exists(os.path.dirname(name)):
       convert_tiff_directory_action(args.source,args.dest,num_imgs)
   else:
       print('ERROR: Verify source and destination paths exists')
       

if __name__ == '__main__':
    main()