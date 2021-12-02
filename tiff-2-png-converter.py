import argparse
import sys
import os 
import cv2
import numpy as np
import imagesize
import math

def convert_tiff_directoryAction(path):
    list_files,width, height = list_folder_files(path)
    depth = len(list_files)
    image_r = np.zeros((width,height,0))
    image_g = np.zeros((width,height,0))
    image_b = np.zeros((width,height,0))
    images = np.zeros((width,height,0))
    for fname in list_files:
        img = cv2.imread(fname)
        if fname.endswith("ch1"):
             print("Load Image " + str(image_r.shape[2]) + " Channel 1 - "+ fname )
             np.vstack((image_r,img))
        elif fname.endswith("ch2"):
             print("Load Image " + str(image_g.shape[2]) + " Channel 2 - "+ fname )
             np.vstack((image_g,img))
        elif fname.endswith("ch3"):
             print("Load Image " + str(image_b.shape[2]) + " Channel 3 - "+ fname )
             np.vstack((image_b,img))
        else:
             print("Load Image " + str(images.shape[2]) + " RGB - "+ fname )
             image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
             np.vstack((images,image_rgb))
             

    if image_r.shape[0] != 0 or  image_g.shape[0] != 0 or image_g.shape[0] != 0:
        merge_RGB(image_r, image_g, image_b, images)

    saveToImage(images,depth,width, height,"test.png")
  
def saveToImage(images,depth,width,height, filename):
    z_slices = depth + 1
    dim = math.ceil(math.sqrt(z_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil( max(width, height) / max_res)
    print("resolution_image_before " + str(width)+ " "+str( height))  
    print("downscale " + str(downscale))  
	
    resolution_image_x,resolution_image_y = width / downscale, height / downscale
    resolution_out = max(width, height) / downscale
    count = 0
    image_out = np.zeros((dim,dim))
    for i in range(dim):
        row_imgs = np.zeros((1,1))
        for j in range(dim):
            tmp_img = np.zeros((resolution_image_x,resolution_image_y))
            if count < images.shape[2] and count <= depth:
               tmp_img =  cv2.resize(images[:,:,i], dim)
            else:
               tmp_img =  cv2.resize(images[:,:,i], dim)
               tmp = np.zeros((resolution_image_x,resolution_image_y))
            if j == 0:
                image_row = np.copy(tmp_img)
            else:
                image_row = cv2.hconcat(image_row,tmp_img)
            count = count+1
        if i == 0:
            image_out = np.copy(image_row)
        else:
            image_out = cv2.vconcat(image_out,image_row)
    status = cv2.imwrite('/home/img/python_grey.png',image_out)
    print("Image written to file-system : ",status)
    

    
def merge_RGB(image_r,  image_g,  image_b, rgb_images, width, heigh, depth):
    zero_image = np.zeros((width,heigh))
    for z in range(depth):
        r_channel = np.zeros((width,heigh))
        g_channel = np.zeros((width,heigh))
        b_channel = np.zeros((width,heigh))
        if image_r.shape[2] != 0:
            r_channel = image_r[:,:,z]
        else:
            r_channel = zero_image
        if image_g.shape[2] != 0:
            g_channel = image_g[:,:,z]
        else:
            g_channel = zero_image
        if image_b.shape[2] != 0:
            b_channel = image_b[:,:,z]
        else:
            b_channel = zero_image
        merged_image = cv2.merge([r_channel,g_channel,b_channel])
        np.vstack(rgb_images,merged_image)


    

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


    return file_list,height,width

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