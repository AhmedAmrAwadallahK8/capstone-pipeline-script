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
         print("Program will now terminate")
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
   clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   img = clahe.apply(img)
   img = cv.copyMakeBorder(img,
                           border_pad,
                           border_pad,
                           border_pad,
                           border_pad,
                           cv.BORDER_CONSTANT,
                           value=0)
   return img

def run_model(x, pipeline):
   pred = pipeline(x)[0]
   np_pred = pred.to_numpy()
   outlines = np_pred[0,1,:,:]
   fills = np_pred[0,0,:,:]
   # fills = cv.resize(fills, dsize=(image_shape[0], image_shape[1]))*255
   return outlines, fills

if __name__ == '__main__':
   inputs = sys.argv
   check_valid_inputs(inputs)
   check_valid_file_type(inputs)

   # Load Model
   model = download_model()

   # Create output folder
   output_folder = "./output"
   if not os.path.exists(output_folder):
      os.makedirs(output_folder)
 
   #Img Info 
   img_path = inputs[1]
   img_name = get_image_name(img_path)
   img_extension = img_path.split(".")[-1]

   # Load Img
   # This part needs to be adjusted to proccess any img from the allowed types
   # into a form that is always the same for the rest of the program
   img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
   img = process_img_type(img, img_extension)
   img = preprocess_image(img)
   color_img = img.copy()
   color_img = cv.cvtColor(color_img, cv.COLOR_GRAY2BGR)

   # Prep Image
   prepared_img = cv.resize(img, dsize=(512, 512), interpolation=cv.INTER_CUBIC)
   prepared_img = prepared_img[np.newaxis, np.newaxis, ...]
   input_array = xr.DataArray(prepared_img, dims=tuple(model.inputs[0].axes))

   # Prep Model
   devices = None
   weight_format = None
   pred_pipeline = bioimageio.core.create_prediction_pipeline(
      model, devices=devices, weight_format=weight_format
   )
   
   # Run Model 
   outlines1, fills1 = run_model(input_array, pred_pipeline) 

   # Execute Second Pass
   prepared_img2 = fills1[np.newaxis, np.newaxis, ...]
   input_array2 = xr.DataArray(prepared_img2, dims=tuple(model.inputs[0].axes))
   outlines2, fills2 = run_model(input_array2, pred_pipeline)

   # Execute Third Pass
   prepared_img3 = fills2[np.newaxis, np.newaxis, ...]
   input_array3 = xr.DataArray(prepared_img3, dims=tuple(model.inputs[0].axes))
   outlines3, fills3 = run_model(input_array3, pred_pipeline)

   # Postprocess Outlines
   # For now using outlines 1, THIS IS TEMPORARY
   final_outlines = cv.resize(
      outlines3, dsize=(img.shape[0], img.shape[1])
   )*255
   final_outlines = final_outlines.astype("uint8")

   blur = cv.GaussianBlur(final_outlines, (5,5), 0)
   _, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

   # Add outlines to image
   outlined_img = img.copy()
   outlined_img = outlined_img.astype("uint8")
   outlined_img[threshold > 0] = 255

   # Save outlines img
   outlines_name = output_folder + "/" + img_name + "_outlines.png"
   fig, ax = plt.subplots(figsize=(16, 16))
   ax.imshow(outlined_img, cmap="gray")
   plt.axis("off")
   plt.savefig(outlines_name, bbox_inches="tight")

   # Contour Hierarchy
   outline_pad = 32
   padded_outlined_img = cv.copyMakeBorder(
      outlined_img,
      outline_pad, outline_pad, outline_pad, outline_pad,
      cv.BORDER_CONSTANT, value=255
   )
   th, threshed = cv.threshold(
      padded_outlined_img, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU
   )
   contours, hierarchy = cv.findContours(
      threshed, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1
   )
   particle_counts = {}
   hierarchy = hierarchy[0] # Removing Redundant Dimension
   curr_parent_id = -1
   for curr_node_id in range(len(hierarchy)):
      parent_node_id = hierarchy[curr_node_id][3]
      if parent_node_id == -1:
         continue
      elif parent_node_id == 0:
         curr_parent_id = curr_node_id
         particle_counts[curr_parent_id] = 0
      else:
         if curr_parent_id in particle_counts.keys():
            particle_counts[curr_parent_id] += 1

   # Figure with particle_ids
   final_output = padded_outlined_img.copy()
   final_output = cv.cvtColor(final_output, cv.COLOR_GRAY2BGR)
   for c_id in particle_counts.keys():
      rect = cv.minAreaRect(contours[c_id])
      box = cv.boxPoints(rect)
      box = np.int0(box)
      final_output = cv.drawContours(final_output, [box], 0, (255, 0, 0), 10)

   for c_id in particle_counts.keys():
      rect = cv.minAreaRect(contours[c_id])
      box = cv.boxPoints(rect)
      box = np.int0(box)
      x1 = box[0, 0]
      y1 = box[0, 1]
      x2 = box[2, 0]
      y2 = box[2, 1]
      rect_center = (int((x1+x2)/2), int((y1+y2)/2))
      text = str(c_id)
      cv.putText(img=final_output,
                 text=text,
                 org=rect_center,
                 fontFace=cv.FONT_HERSHEY_TRIPLEX,
                 fontScale=0.85,
                 color=(0,255,0),
                 thickness=2)
   
   file_name = output_folder + "/" + img_name + "_labeled_contours.png"
   fig, ax = plt.subplots(figsize=(16, 16))
   ax.imshow(final_output, cmap="gray")
   plt.axis("off")
   plt.savefig(file_name, bbox_inches="tight")

   # csv of particle counts
   file_name = output_folder + "/" + img_name + "_particle_counts.csv"
   csv_file = open(file_name, "w")
   csv_file.write("id,counts\n")
   for id in particle_counts.keys():
      counts = particle_counts[id]
      observation = str(id) + "," + str(counts) + "\n"
      csv_file.write(observation) 

   csv_file.write("\n")
   csv_file.close()
   print("Script Finished Succesfully")













