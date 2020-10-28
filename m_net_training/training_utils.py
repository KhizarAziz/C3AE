import numpy as np


# adding random box in image with random colored pixels, it makes model generic????
def random_erasing(img, drop_out=0.3, aspect=(0.5, 2), area=(0.06, 0.10)):
    # https://arxiv.org/pdf/1708.04896.pdf
    if 1 - random.random() > drop_out:
        return img
    img = img.copy()
    height, width = img.shape[:-1]
    aspect_ratio = np.random.uniform(*aspect)
    area_ratio = np.random.uniform(*area)
    img_area = height * width * area_ratio
    dwidth, dheight = np.sqrt(img_area * aspect_ratio), np.sqrt(img_area * 1 / aspect_ratio) 
    xmin = random.randint(0, height)
    ymin = random.randint(0, width)
    xmax, ymax = min(height, int(xmin + dheight)), min(width, int(ymin + dwidth))
    img[xmin:xmax,ymin:ymax,:] = np.random.random_integers(0, 256, (xmax-xmin, ymax-ymin, 3))
    return img



def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, category)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < category:
            embed[idx+1] = right_prob
        return embed
    return np.array(age_split(age_label))



def image_transform(row,target_img_shape=(64,64),dropout=0.,require_augmentation):
  # read image from buffer then decode
  img = np.frombuffer(row["image"], np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  #add random noise
  if require_augmentation:
    img = random_erasing(img,dropout=dropout)
  #add padding, incase any face location is negative (face is not full)
  padding = 50
  img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
  # get trible box (out,middle,inner) and crop image from these boxes then
  tripple_cropped_imgs = []
  for box in pickle.loads(row['triblebox'],encoding="bytes"): # deserializing object which we converted to binary format using myNumArray.dump() method
    h_min, w_min = bbox[0] # xmin,ymin
    h_max, w_max = bbox[1] #xmax, ymax
    # crop image according to box size and add to list
    img = new_bd_img[w_min+padding:w_max+padding, h_min+padding: h_max+padding] # cropping image
    img = cv2.resize(img, (64,64)) # resize according to size we want
    tripple_cropped_imgs.append(img)
    # image augmentaion (hue, contrast,rotation etc) if needed
    if require_augmentation:
       flag = random.randint(0, 3)
       contrast = random.uniform(0.5, 2.5)
       bright = random.uniform(-50, 50)
       rotation = random.randint(-15, 15)
       cascad_imgs = [image_enforcing(x, flag, contrast, bright, rotation) for x in tripple_cropped_imgs]
       
  return cascad_imgs    


def img_and_age_data_generator(dataset_df, batch_size=32,augmentation):
  dataset_df = dataset_df.reset_index(drop=True)
  df_count = len(dataset_df)
  idx = np.random.permutation(df_count) # it will return a list of numbrs (0-df_count), in randomnly arranged
  start = 0
  while start+batch_size < df_count:
    idx_to_get = idx[start+batch_size] # making a list of random indexes, to get them from dataset
    current_batch = dataset_df.iloc[idx_to_get] # fetching some list, which is our batch

    #load imgs, transform& create a list
    img_List = []
    two_point_ages = [] # list for 2_point_rep of ages
    for index,row in current_batch.iterrows(): #iterate over batch to load & transform each img
      # load and transform image
      img = image_transform(row, is_training=is_training, dropout=dropout,require_augmentation=augmentation)
      img_List.append(img)
      # make 2_point_represenation(list) of age
      two_point_rep = two_point(int(row.age), 12, 10)
      two_point_ages.append(two_point_rep)    

    img_nparray = np.array(img_List) # converting image list to np
    two_point_ages_nparray = np.array(two_point_ages) # converting to np
    out = [current_batch.age.tonumpy,two_point_ages_nparray] # making list of age_array & 2point_reprseation_array

    yield [img_nparray[:,0], img_nparray[:,1], img_nparray[:,2]], out # return batch
    start += batch_size # update start point, for next batch




