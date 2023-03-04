import os
import sys
import bioimageio.core
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cv2 as cv

valid_extensions = ["jpg", "png", "tiff", "tif"]
negative_response = ["no", "n"]
positive_response = ["yes", "y"]

USER_EXIT = 0

class InvalidFileTypeException(Exception):
   def __init__(self):
      msg = "This script only accepts images of the following type "
      msg += str(valid_extensions)
      super().__init__(msg)

class UnsupportedFileTypeException(Exception):
   def __init__(self):
      msg = "The image entered is flagged as a valid type but is currently "
      msg += "unsupported. This is an error with the script itself and not "
      msg += "the user unless the user changed the script."
      super().__init__(msg)

class MissingInputException(Exception):
   def __init__(self):
      msg = "This script expects exactly 1 input but nothing was input. "
      msg += "Rerun the script with the image file you want to process."
      super().__init__(msg)

class ExcessiveInputException(Exception):
   def __init__(self):
      msg = "This script expects exactly 1 input but more than one was "
      msg += "input with the script. Rerun the script with exactly one "
      msg += "file path"
      super().__init__(msg)

def check_valid_inputs(inputs):
   if len(inputs) < 2:
      raise MissingInputException
   if len(inputs) > 2:
      raise ExcessiveInputException

def check_valid_file_type(inputs):
   data_direc = inputs[1]
   extension = data_direc.split(".")[-1]
   if extension not in valid_extensions:
      raise InvalidFileTypeException

def get_image_name(img_path):
   split_path = img_path.split("/")
   img_file = split_path[-1]
   img_name = img_file.split(".")[0]
   return img_name

def process_img_type(img, ext):
   if ext == "jpg":
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      return img
   elif ext == "png":
      img = img[:,:,:3]
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      return img
   elif ext == "tif":
      return img
   elif ext == "tiff":
      return img
   else:
      raise InvalidFileTypeException

def download_model():
   print(""" This script needs to install an AI model from the internet before
it can function. The total size is around 200MB. If you are not okay with the 
download enter "no" below otherwise enter "yes" """)
   invalid_entry = True
   while(invalid_entry):
      user_response = input("Proceed with Download? ").lower()
      if user_response in positive_response:
         invalid_entry = False
         rdf_doi = "10.5281/zenodo.5764892"
         model_resource = bioimageio.core.load_resource_description(rdf_doi)
         return model_resource
      elif user_response in negative_response:
         invalid_entry = False
         print("Program wll now terminate")
         sys.exit(USER_EXIT)
      else:
         print(""" Invalid input. Enter either "yes" or "no". """)

def rescale_image(img):
   img_scaled = img.copy()
   if np.amax(img) > 255:
      img_scaled = cv.convertScaleAbs(img, alpha = (255.0/np.amax(img)))
   else:
      img_scaled = cv.convertScaleAbs(img)
   return img_scaled

def preprocess_image(img):
   img = rescale_image(img)
   border_pad = 64
   img = cv.copyMakeBorder(img,
                           border_pad,
                           border_pad,
                           border_pad,
                           border_pad,
                           cv.BORDER_CONSTANT,
                           value=0)
   clahe = cv.createClahe(clipLimit=2.0, tileGridSize=(8,8))
   img = clahe.apply(img)
   return img


if __name__ == '__main__':
   inputs = sys.argv
   check_valid_inputs(inputs)
   check_valid_file_type(inputs)

   # Load Model
   model = download_model()
  
   img_path = inputs[1]
   img_name = get_image_name(img_path)
   img_extension = img_path.split(".")[-1]

   # Load Img
   # This part needs to be adjusted to proccess any img from the allowed types
   # into a form that is always the same for the rest of the program
   img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
   img = process_img_type(img, img_extension)
   color_img = img.copy()
   color_img = cv.cvtColor(color_img, cv.COLOR_GRAY2BGR)
   


   print("Script Finished Succesfully")













