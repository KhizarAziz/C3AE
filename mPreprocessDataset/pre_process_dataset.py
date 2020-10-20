
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import load_mat


# load dataset
# process it to detect faces, detect landmarks, align, & make 3 sub boxes which will be used in next step to feed into network
# save dataset as pandas,feather & imencode for size efficiency


DataFrame = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"])

Class Process_WIKI():
  
  def __init__(meta_file_path,):
    print('wiki init')
    #initialize all parameters of dataset
    self.meta_file = load_mat(meta_file_path)

  def meta_to_csv():
    self.meta_file = 

if __name__ == "__main__":
  
  # define all parameters here
  dataset_name = 'wiki' # different dataset name means different sequence for loading etc

  if dataset_name == 'wiki':
    #init wiki object with initial parameters
    # call wiki convert meta.mat to df.csv
    # load all images
    # call process all images
  elif dataset_name == 'imdb':
    #init imdb object with initial parameters
    # call imdb convert meta.mat to df.csv
    # load all images
    # call process all images



