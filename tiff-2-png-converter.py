
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



class Tiff_Processor():
    verbose = False
    equalize_histogram = False
    equalize_histogram_method = None
    gamma = False
    gamma_val = 1
    num_imgs = None
    source = None
    dest = None

    @staticmethod
    def add_arguments(parser):
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
     
    @staticmethod
    def configure(args):

        name, extension = os.path.splitext(args.dest)
        if os.path.exists(args.source) and os.path.exists(os.path.dirname(name)):
            Tiff_Processor.source = args.source
            Tiff_Processor.dest = name
        else:
            print('ERROR: Verify source and destination paths exists')
            sys.exit()

        if args.verbose:
            Tiff_Processor.verbose = True
        if args.equalize1 or args.equalize2:
            Tiff_Processor.equalize_histogram = True
            if args.equalize1:
                Tiff_Processor.equalize_histogram_method = equalize_image_histogram_opencv
            if args.equalize2:
                Tiff_Processor.equalize_histogram_method = equalize_image_histogram_custom
        
        if args.gamma:
            Tiff_Processor.gamma = True
            Tiff_Processor.gamma_val = args.gamma
        if args.num_imgs:
            Tiff_Processor.num_imgs = int(args.num_imgs)
    
     
    def convert_to_png(self):
        list_files,width, heigth = self.list_folder_files(self.source)
        if self.num_imgs is None:
            self.num_images = len(list_files)
        image_r = [None] * self.num_imgs
        image_g = [None] * self.num_imgs
        image_b = [None] * self.num_imgs
        images = [None] * self.num_imgs
        bits_per_sample = ""
        printProgressBar(0, self.num_imgs, prefix = 'grouping channels:', suffix = 'Complete', length = 50)
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
            if "ch1" in name:
                if self.verbose:
                    print("Load Image " + str(sequence_number_int) + " Channel 1 - "+ file_full_path )
                image_r[sequence_number_int - 1] = img
            elif "ch2" in name:
                if self.verbose:
                    print("Load Image " + str(sequence_number_int) + " Channel 2 - "+ file_full_path )
                image_g[sequence_number_int - 1] = img
            elif "ch3" in name:
                if self.verbose:
                    print("Load Image " + str(sequence_number_int) + " Channel 3 - "+ file_full_path )
                image_b[sequence_number_int - 1] = img
            else:
                if self.verbose:
                    print("Load Image " + str(sequence_number_int) + " RGB - "+ file_full_path )
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
                images[sequence_number_int - 1] = image_rgb
            printProgressBar(i + 1, self.num_imgs, prefix = 'grouping channels:', suffix = 'Complete', length = 50)    


        if len(image_r) != 0 or len(image_g) != 0 or len(image_b) != 0:
            self.merge_RGB(image_r, image_g, image_b, images,width,heigth,bits_per_sample)

        print("##Saving Image to: "+ self.dest)
        self.save_to_Image(images,width, heigth)

    def interpolateData(self,img_array, width,heigth, num_images):
        ## STILL IN DEVELOP - NOT SURE IF WE NEED IT
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

    def save_to_Image(self,volume,width,height):

        z_slices = self.num_imgs + 1
        dim = math.ceil(math.sqrt(z_slices))
        max_res = math.floor(4096 / dim)
        downscale = math.ceil( max(width, height) / max_res)
        if self.verbose:
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
                if count < len(volume) and count <= self.num_imgs:
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
        
        name, extension = os.path.splitext(self.dest)
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
        file.write("Depth:"+str(self.num_imgs)+"\n")
        file.write("bps:"+str(image_out.dtype)+"\n")
        file.write("amax:"+str(amax)+"\n")
        file.close()
        print("Image written to file-system : "+name+extension)
        
        
    def merge_RGB(self,image_r,  image_g,  image_b, rgb_images, width, heigh,bits_per_sample):
        zero_image = np.zeros((width,heigh),dtype=bits_per_sample)
        printProgressBar(0, self.num_imgs, prefix = 'Mergin channels:', suffix = 'Complete', length = 50)
        for z in range(self.num_imgs):
            r_channel = np.zeros((width,heigh),dtype=bits_per_sample)
            g_channel = np.zeros((width,heigh),dtype=bits_per_sample)
            b_channel = np.zeros((width,heigh),dtype=bits_per_sample)
            if image_r[z] is not None:
                r_channel = image_r[z]
                if self.equalize_histogram:
                    r_channel = self.equalize_histogram_method(r_channel)
            else:
                r_channel = zero_image
            if image_g[z] is not None:
                g_channel = image_g[z]
                if self.equalize_histogram:
                    g_channel = self.equalize_histogram_method(g_channel)
            else:
                g_channel = zero_image
            if image_b[z] is not None:
                b_channel = image_b[z]
                if self.equalize_histogram:
                    b_channel = self.equalize_histogram_method(b_channel)
            else:
                b_channel = zero_image
            

            merged_image = cv2.merge([r_channel,g_channel,b_channel])
            
            if self.gamma:
                float_gamma = float(self.gamma_val)
                merged_image = gamma_correction(merged_image,float_gamma)
            rgb_images[z]=  merged_image

            printProgressBar(z+1, self.num_imgs, prefix = 'Mergin channels:', suffix = 'Complete', length = 50)
        
    def list_folder_files(self,path_to_foder):

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
   # Create argument parser
   parser = argparse.ArgumentParser()
   # Add arguments
   Tiff_Processor.add_arguments(parser)
   args = parser.parse_args()
   Tiff_Processor.configure(args)
   tiff_processor = Tiff_Processor()
   tiff_processor.convert_to_png()
   
       

if __name__ == '__main__':
    main()