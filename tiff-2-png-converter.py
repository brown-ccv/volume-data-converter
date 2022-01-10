
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

verbose = False
equalize_histogram = False
gamma = False
gamma_val = 1


def convert_tiff_directory_action(source_path, dest_path):
    list_files,width, heigth = list_folder_files(source_path)
    num_images = len(list_files)
    image_r = [None] * num_images
    image_g = [None] * num_images
    image_b = [None] * num_images
    images = [None] * num_images
    bits_per_sample = ""
    printProgressBar(0, num_images, prefix = 'grouping channels:', suffix = 'Complete', length = 50)
    for i in range(num_images):
        file_full_path = list_files[i]
        img = tiff.imread(list_files[i])
        if bits_per_sample=="":
            bits_per_sample = img.dtype
        elif bits_per_sample!=img.dtype:
            print("image " + file_full_path + "has a different sample type "+bits_per_sample+" expected: "+bits_per_sample)
            sys.exit("Check images bits per sample tag")
        name, extension = os.path.splitext(file_full_path)
        if "ch1" in name:
             if verbose:
                print("Load Image " + str(i) + " Channel 1 - "+ file_full_path )
             image_r[i] = img
        elif "ch2" in name:
             if verbose:
                print("Load Image " + str(i) + " Channel 2 - "+ file_full_path )
             image_g[i] = img
        elif "ch3" in name:
             if verbose:
                print("Load Image " + str(i) + " Channel 3 - "+ file_full_path )
             image_b[i] = img
        else:
             if verbose:
                print("Load Image " + str(i) + " RGB - "+ file_full_path )
             image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
             images[i] = image_rgb
        printProgressBar(i + 1, num_images, prefix = 'grouping channels:', suffix = 'Complete', length = 50)
             

    if len(image_r) != 0 or len(image_g) != 0 or len(image_b) != 0:
        merge_RGB(image_r, image_g, image_b, images,width,heigth,num_images,bits_per_sample)

    print("##Saving Image to: "+dest_path)
    save_to_Image(images,num_images,width, heigth,dest_path)
  
def save_to_Image(volume,depth,width,height, dest_path):

    # scale data to 0.0-1.0
    # for i in range(len(volume)):
    #     data = volume[i].copy()
    #     max_data = np.amax(data)
    #     min_data = np.amin(data)
    #     data = (data - min_data)/(max_data - min_data)
    #     data = data * 255
    #     volume[i] = data.astype('uint16')

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
    
    
    #src = cv2.cvtColor(src, cv.COLOR_BGR2GRAY)
    #image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2GRAY)
    #image_out_8bit = image_out.astype(np.uint8)
    amax = np.amax(image_out)    
    # hist,bins = np.histogram(image_out.flatten(),amax,[0,amax])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    #normalized_image = cv2.equalizeHist(image_out_8bit)
    
    #resultimage = np.zeros((height,width))
    #normalized_image= cv2.normalize(img,  resultimage, 0, 255,cv2.NORM_MINMAX)
    # print("image_out.amax: "+str(amax))
    # print("image_out.dtype: "+str(image_out.dtype))
    # normalized_image = np.array((image_out.shape),dtype="uint16")
    # print("normalized_image.dtype: "+str(normalized_image.dtype))
    # normalized_image = image_out // np.uint16(amax)
    # print("normalized_image.dtype2: "+str(normalized_image.dtype))
    # print(np.amax(normalized_image))
    # normalized_image = normalized_image*np.uint16(255)
    # print(np.amax(normalized_image))

    #amax2 = np.amax(normalized_image)
    #print("normalized_image.dtype: "+str(normalized_image.dtype))
    #status = cv2.imwrite(name+extension,image_out)
    #normalized_image_u16bit = normalized_image.astype(np.uint16)
    norm = np.zeros((image_out.shape))
    normalized_image_u16bit = cv2.normalize(image_out,norm,0,65535,cv2.NORM_MINMAX)
    numpngw.write_png(name+extension,image_out)
    numpngw.write_png(name+"_normalized"+extension,normalized_image_u16bit)
    
    

    f = open(name+"_metadata", "a")
    f.write("Width:"+str(width)+"\n")
    f.write("Heigth:"+str(height)+"\n")
    f.write("Depth:"+str(depth)+"\n")
    f.write("bps:"+str(image_out.dtype)+"\n")
    f.write("amax:"+str(amax)+"\n")
    #f.write("amax normalized:"+str(amax2)+"\n")
    f.close()
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
                r_channel = equalize_imgage_histogram(r_channel)
        else:
            r_channel = zero_image
        if image_g[z] is not None:
            g_channel = image_g[z]
            if equalize_histogram:
                g_channel = equalize_imgage_histogram(g_channel)
        else:
            g_channel = zero_image
        if image_b[z] is not None:
            b_channel = image_b[z]
            if equalize_histogram:
                b_channel = equalize_imgage_histogram(b_channel)
        else:
            b_channel = zero_image
        #merged_image = np.dstack([r_channel,g_channel,b_channel])

        merged_image = cv2.merge([r_channel,g_channel,b_channel])
        #merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB )
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
    
   parser.add_argument("--equalize",
                        help="equilize histogram in volume slices",action='store_true')
   
   parser.add_argument("--gamma",
                        help="Apply gamma correction")
                        
   args = parser.parse_args()
   name, extension = os.path.splitext(args.dest)
   if args.verbose:
       global verbose
       verbose = True

   if args.equalize:
       global equalize_histogram
       equalize_histogram = True
    
   if args.gamma:
       global gamma,gamma_val
       gamma = True
       gamma_val = args.gamma

   if os.path.exists(args.source) and os.path.exists(os.path.dirname(name)):
       convert_tiff_directory_action(args.source,args.dest)
   else:
       print('ERROR: Verify source and destination paths exists')
       

if __name__ == '__main__':
    main()