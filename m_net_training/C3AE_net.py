from m_net_training import training_utils

def preprocessing(dataframes,batch_size = 32, category=12, interval=10,input_imgs_shape =(64,64), augmentation=True, dropout = 0.2):
  # category: bin + 2 due to two side
  # interval: age interval
  import imp
  imp.reload(training_utils)
  # return training_utils.img_and_age_data_generator(dataset_df=dataframes,category=category,interval=interval,imgs_shape=input_shape, batch_size=batch_size,augmentation=augmentation,dropout=dropout)
  
