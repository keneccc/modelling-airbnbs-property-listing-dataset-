import os 

import os, glob
from PIL import Image 
import cv2
class prepare_image():

    def download_images(self):


        image_folder = "images/"
        processed_images_folder="processed_images/"


        for root,dir,files in os.walk(image_folder):
            for file in files:
                
            
                filepath = os.path.join(root,file) 
                #print(filepath)
                img = Image.open(filepath)
                

                width, height = img.size
                new_height = 500
                new_width = 500


                resized_img = img.resize((new_width, new_height))
                
                #resized_img.save(processed_images_folder + file, resized_img)
                # new_filename = os.path.join(processed_images_folder, os.path.basename(filename))

                target_location = os.path.join(processed_images_folder,dir[0])
                if not os.path.exists( target_location ):
                    os.makedirs(target_location)
                    resized_img.save(target_location + file +"_processed.png")
                    #print(filename)}
