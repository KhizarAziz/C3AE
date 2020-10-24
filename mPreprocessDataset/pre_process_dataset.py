import pandas as pd
import numpy as np
import cv2
import json # to serialize objects, so can be stored as string in pandas's feather file
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from datetime import datetime
import dlib
from pose import get_rotation_angle

# process it to detect faces, detect landmarks, align, & make 3 sub boxes which will be used in next step to feed into network
# save dataset as pandas,feather & imencode for size efficiency

Dataset_DF = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"])
#initiate face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")

def gen_boundbox(box, landmark):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    ymin, xmin, ymax, xmax = box.bottom(), box.left(), box.top(), box.right()
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark.parts()[30].x, landmark.parts()[30].y) # calculating nose center point, so the triple boxes will be cropped according to nose point
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # Contains the smallest frame
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])


def calculate_age(dob, image_capture_date):
  birth = datetime.fromordinal(max(int(dob) - 366, 1))
  # assume the photo was taken in the middle of the year
  if birth.month < 7:
      return image_capture_date - birth.year
  else:
      return image_capture_date - birth.year - 1



class Process_WIKI_IMDB():
  
  def __init__(self,base_path,dataset_name,extra_padding):
    #init meta.mat file
    self.dataset_name = dataset_name
    meta_file_name = dataset_name+'.mat'
    self.base_path = Path(base_path)
    self.meta_file_path = self.base_path.joinpath(meta_file_name)

    self.extra_padding = extra_padding

  def meta_to_csv(self,dataset_name):
    test_num = 10
    # make list of all columns in meta.mat file
    meta = loadmat(self.meta_file_path)
    full_path = [p[0] for p in meta[dataset_name][0, 0]["full_path"][0][:test_num]]
    dob = meta[dataset_name][0, 0]["dob"][0][:test_num]  # Matlab serial date number
    mat_gender = meta[dataset_name][0, 0]["gender"][0][:test_num]
    photo_taken = meta[dataset_name][0, 0]["photo_taken"][0][:test_num]  # year
    face_score = meta[dataset_name][0, 0]["face_score"][0][:test_num]
    second_face_score = meta[dataset_name][0, 0]["second_face_score"][0][:test_num]
    age = [calculate_age(dob[i],photo_taken[i] ) for i in range(len(dob))] # calculate age using dob and taken date
    # make a dataframe dataset
    Dataset_DF = pd.DataFrame({"full_path": full_path, "age": age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
    # Completing image paths from root to image
    Dataset_DF['full_path'] = str(self.base_path)+'/'+Dataset_DF['full_path']
    # save dataframe as csv
    Dataset_DF.to_csv(self.base_path.joinpath(self.dataset_name+'.csv'),index=False)

  def detect_faces_and_landmarks(self,image):
    face_rect_list = detector(image,1)
    # make a landmarks_list of all faces detected in image
    lmarks_list = dlib.full_object_detections()
    for face_rect in face_rect_list:
      lmarks_list.append(predictor(image, face_rect)) # getting landmarks as a list of objects
    return face_rect_list, lmarks_list

  def crop_and_transform_images(self):
    meta_dataframe = pd.read_csv(self.base_path.joinpath(self.dataset_name+'.csv'))
    #filter out where second face != null (image have 2 faces)
    meta_dataframe = meta_dataframe[meta_dataframe.second_face_score.isna()]
  
    # init lists of all properties gonna be saved
    properties_list = []
    # loop through meta.csv for all images
    for index,series in meta_dataframe.iterrows():
      # clear multiple faces
      image_path = series.full_path
      # try:
      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      image = cv2.copyMakeBorder(image, self.extra_padding, self.extra_padding, self.extra_padding, self.extra_padding, cv2.BORDER_CONSTANT)
      face_rect_list, lmarks_list = self.detect_faces_and_landmarks(image) # Detect face & landmarks
      if len(face_rect_list) > 1:
        raise Exception("more than 1 faces in image",image_path )
        continue
      elif len(face_rect_list) < 1:
        raise Exception("No Face found in image ",image_path)
        continue
      #extract_image_chips will crop faces from image according to size & padding and align them in upright position and return list of them
      cropped_faces = dlib.get_face_chips(image, lmarks_list,size=64, padding=0.4)  # aligned face with padding 0.4 in papper
      # detect face landmarks again from cropped & align face.  (as positions of lmarks are changed in cropped image)
      face_rect_list, lmarks_list = self.detect_faces_and_landmarks(cropped_faces[0]) # Detect face from cropped image
      face_rect, first_lmarks = face_rect_list[0], lmarks_list[0] # getting first face's rectangle box and landmarks 
      trible_box = gen_boundbox(face_rect, first_lmarks) # get 3 face boxes for nput into network, as reauired in paper
      pitch, yaw, roll = get_rotation_angle(face_rect_list[0], first_lmarks) # gen face rotation for filtering
      image = face_rect_list[0] # select the first align face and replace
      # except Exception as ee:
      #     print('exption ',ee)
      #     trible_box = np.array([])
      #     face_rect, first_lmarks = np.array([]), np.array([])
      #     pitch, yaw, roll = np.nan, np.nan, np.nan
      #     age = np.nan
      #     gender = np.nan
      
      status, buf = cv2.imencode(".jpg", image)
      image_buffer = buf.tostring()
    #   face_rect_serialized = json.dumps(face_rect,indent = 2)  # xmin, ymin, xmax, ymax
    #   trible_boxes_serialized =json.dumps(trible_box,indent = 2) # 3 boxes of face as required in paper
    #   face_yaw = yaw
    #   face_pitch = pitch
    #   face_roll = roll
    #   # converting landmarks (face_detection_object) to array so can be converted to json
    #   landmarks_array = []
    #   for point in lm.parts():
    #     landmarks_array.append([point.x,point.y])
    #   face_landmarks_serialized = json.dumps(landmarks_array,indent = 2)  # y1..y5, x1..x5

    #   # adding determined values to properties list so that can later be converted to DF -> CSV
    #   properties_list.append([image_path,image_buffer,face_rect_serialized,trible_boxes_serialized,face_yaw,face_pitch,face_roll,face_landmarks_serialized])
      
    # print(len(properties_list),len(meta_dataframe))

    # new_DataFrame = pd.DataFrame(properties_list,columns=['image_path','image','org_box','trible_box','yaw','pitch','roll','landmarks'])
    # new_DataFrame.to_csv('/content/Dataset.csv')
    # return new_DataFrame


if __name__ == "__main__":
  # define all parameters here
  dataset_directory_path = '/content/C3AE/dataset/wiki_crop'
  dataset_name = 'wiki' # different dataset name means different sequence for loading etc

  # image transform params (if require)
  extra_padding = 0

  if dataset_name == 'wiki' or dataset_name == 'imdb': # because structure is same
    dataset_class_ref_object = Process_WIKI_IMDB(dataset_directory_path,dataset_name,extra_padding)
    dataset_class_ref_object.meta_to_csv(dataset_name) # convert meta.mat to meta.csv
    dataset_class_ref_object.crop_and_transform_images()
    # call wiki convert meta.mat to df.csv
    # load all images
    # call process all images
  elif dataset_name == 'imdb':
    print('imdb')
    #init imdb object with initial parameters
    # call imdb convert meta.mat to df.csv
    # load all images
    # call process all images
