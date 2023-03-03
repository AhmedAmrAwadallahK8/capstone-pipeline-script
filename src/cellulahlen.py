import os
import sys
import bioimageio.core
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cv2 as cv

valid_extensions = ["jpg", "png"]

class InvalidFileTypeException(Exception):
   def __init__(self):
      msg = " This script only accepts images of the following type "
      msg += str(valid_extensions)
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



if __name__ == '__main__':
   inputs = sys.argv
   check_valid_inputs(inputs)
   check_valid_file_type(inputs)
  
   img_path = inputs[1]
   img_name = get_image_name(img_path)
   img_extension = img_path.split(".")[-1]

   # Load Img
   img = cv.imread(img_path, cv.IMREAD_UNCHANGED)













