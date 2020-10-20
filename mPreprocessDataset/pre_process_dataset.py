
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from datetime import datetime


# load dataset
# process it to detect faces, detect landmarks, align, & make 3 sub boxes which will be used in next step to feed into network
# save dataset as pandas,feather & imencode for size efficiency

Dataset_DF = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"])

class Process_WIKI_IMDB():
  
  def __init__(self,base_path,dataset_name):
    #init meta.mat file
    self.dataset_name = dataset_name
    meta_file_name = dataset_name+'.mat'
    self.base_path = Path(base_path)
    self.meta_file_path = self.base_path.joinpath(meta_file_name)

  def meta_to_csv(self,dataset_name):
    # make list of all columns in meta.mat file
    meta = loadmat(self.meta_file_path)
    full_path = [p[0] for p in meta[dataset_name][0, 0]["full_path"][0][:-1]]
    dob = meta[dataset_name][0, 0]["dob"][0][:-1]  # Matlab serial date number
    mat_gender = meta[dataset_name][0, 0]["gender"][0][:-1]
    photo_taken = meta[dataset_name][0, 0]["photo_taken"][0][:-1]  # year
    face_score = meta[dataset_name][0, 0]["face_score"][0][:-1]
    second_face_score = meta[dataset_name][0, 0]["second_face_score"][0][:-1]
    age = [self.calculate_age(dob[i],photo_taken[i] ) for i in range(len(dob))] # calculate age using dob and taken date
    # make a dataframe dataset
    Dataset_DF = pd.DataFrame({"full_path": full_path, "age": age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
    # save dataframe as csv
    Dataset_DF.to_csv(self.base_path.joinpath(self.dataset_name+'.csv'),index=False)


  def crop_and_transform_images(self, dataframe):

    for index,row in dataframe.iterrow():
      # clear multiple faces
      image_path = series.full_path
      try:
          print(image_path)
          image = cv2.imread(image_path, cv2.IMREAD_COLOR)
          if self.extra_padding:
              extra_padding = self.extra_padding
              image = cv2.copyMakeBorder(image, extra_padding, extra_padding, extra_padding, extra_padding, cv2.BORDER_CONSTANT)
          if not np.isnan(series.second_face_score):
              raise Exception("secend face is not None~---%s~-%s- %s"%(series.name, series.age, image_path))
          bounds, lmarks = gen_face(detector, image, image_path)
          crops = detector.extract_image_chips(image, lmarks, padding=0.4)  # aligned face with padding 0.4 in papper
          if len(crops) == 0:
              print("cant detect!!!!")
              raise Exception("no crops~~ %s---%s"%(image_path, series.age))
          if len(crops) > 1:
              raise Exception("more than one face~---%s~-- %s"%(series.name, image_path))
          bounds, lmarks = gen_face(detector, crops[0], image_path)  # recaculate landmar
          org_box, first_lmarks = bounds[0], lmarks[0]
          trible_box = gen_boundbox(org_box, first_lmarks)
          pitch, yaw, roll = get_rotation_angle(crops[0], first_lmarks) # gen face rotation for filtering
          image = crops[0]   # select the first align face and replace
      except Exception as ee:
          logging.info("exception as ee: %s"%ee)
          print(ee)
          trible_box = np.array([])
          org_box, first_lmarks = np.array([]), np.array([])
          pitch, yaw, roll = np.nan, np.nan, np.nan
          age = np.nan
          gender = np.nan
      status, buf = cv2.imencode(".jpg", image)
      series["image"] = buf.tostring() 
      series["org_box"] = org_box.dumps()  # xmin, ymin, xmax, ymax
      series["landmarks"] = first_lmarks.dumps()  # y1..y5, x1..x5
      series["trible_box"] = trible_box.dumps() 
      series["yaw"] = yaw
      series["pitch"] = pitch
      series["roll"] = roll

      return series

  def calculate_age(self,dob, image_capture_date):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return image_capture_date - birth.year
    else:
        return image_capture_date - birth.year - 1

if __name__ == "__main__":
  # define all parameters here
  dataset_directory_path = '/content/C3AE/dataset/wiki_crop'
  dataset_name = 'wiki' # different dataset name means different sequence for loading etc
  if dataset_name == 'wiki' or dataset_name == 'imdb': # because structure is same
    dataset_class_ref_object = Process_WIKI_IMDB(dataset_directory_path,dataset_name)
    dataset_class_ref_object.meta_to_csv(dataset_name) # convert meta.mat to meta.csv
    # call wiki convert meta.mat to df.csv
    # load all images
    # call process all images
  elif dataset_name == 'imdb':
    print('imdb')
    #init imdb object with initial parameters
    # call imdb convert meta.mat to df.csv
    # load all images
    # call process all images
