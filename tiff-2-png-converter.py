import argparse
import sys
import os 
import cv2
import numpy as np
import imagesize
import math

def convert_tiff_directoryAction(source_path, dest_path):
    list_files,width, heigth = list_folder_files(source_path)
    num_images = len(list_files)
    image_r = [None] * num_images
    image_g = [None] * num_images
    image_b = [None] * num_images
    images = [None] * num_images
    for i in range(num_images):
        file_full_path = list_files[i]
        img = cv2.imread(list_files[i])
        name, extension = os.path.splitext(file_full_path)
        if "ch1" in name:
             print("Load Image " + str(i) + " Channel 1 - "+ file_full_path )
             image_r[i] = img
        elif "ch2" in name:
             print("Load Image " + str(i) + " Channel 2 - "+ file_full_path )
             image_g[i] = img
        elif "ch3" in name:
             print("Load Image " + str(i) + " Channel 3 - "+ file_full_path )
             image_b[i] = img
        else:
             print("Load Image " + str(i) + " RGB - "+ file_full_path )
             image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
             images[i] = image_rgb
             

    if len(image_r) != 0 or len(image_g) != 0 or len(image_b) != 0:
        merge_RGB(image_r, image_g, image_b, images,width,heigth,num_images)

    saveToImage(images,num_images,width, heigth,dest_path)
  
def saveToImage(volume,depth,width,height, dest_path):
    z_slices = depth + 1
    dim = math.ceil(math.sqrt(z_slices))
    max_res = math.floor(4096 / dim)
    downscale = math.ceil( max(width, height) / max_res)
    print("resolution_image_before " + str(width)+ " "+str( height))  
    print("downscale " + str(downscale))  
	
    resolution_image = ( int(width / downscale), int(height / downscale))
    
    count = 0
    image_out = np.zeros(())
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
    name, extension = os.path.splitext(dest_path)
    if not extension:
        extension = ".png"
      
    status = cv2.imwrite(name+extension,image_out)
    f = open(name+"_metadata", "a")
    f.write("Width:"+str(width)+"\n")
    f.write("Heigth:"+str(height)+"\n")
    f.write("Depth:"+str(depth)+"\n")
    f.close()
    print("Image written to file-system : ",status)
    

    
def merge_RGB(image_r,  image_g,  image_b, rgb_images, width, heigh, depth):
    zero_image = np.zeros((width,heigh))
    for z in range(depth):
        r_channel = np.zeros((width,heigh))
        g_channel = np.zeros((width,heigh))
        b_channel = np.zeros((width,heigh))
        if image_r[z] is not None:
            tmp = image_r[z]
            r_channel = tmp[:,:,0]
        else:
            r_channel = zero_image
        if image_g[z] is not None:
            tmp = image_g[z]
            g_channel = tmp[:,:,1]
        else:
            g_channel = zero_image
        if image_b[z] is not None:
            tmp = image_b[z]
            b_channel = tmp[:,:,2]
        else:
            b_channel = zero_image
        merged_image = cv2.merge((r_channel,g_channel,b_channel))
        rgb_images.append(merged_image)


    

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
                        
   args = parser.parse_args()
   name, extension = os.path.splitext(args.dest)
   print(name)
   if os.path.exists(args.source) and os.path.exists(os.path.dirname(name)):
       convert_tiff_directoryAction(args.source,args.dest)
   else:
       print('ERROR: Verify source and destination paths exists')
       

if __name__ == '__main__':
    main()