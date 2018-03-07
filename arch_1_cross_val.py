import pandas as pd
import os
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.model_selection import KFold
import random
import numpy as np
import tensorflow as tf
import time
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization ,Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

if tf.test.gpu_device_name():
    print("GPU device found : {} ".format(tf.test.gpu_device_name()))
else :
    print("No GPU device found, CPU will be used")

data = pd.read_csv('driver_imgs_list.csv')
data["file_path"] = '../imgs/train/'+data["classname"] + '/'+ data["img"]

driver_to_img = dict()
for each_driver in data.subject.unique():
    subset_df = data[data["subject"] == each_driver]
    driver_to_img[each_driver] = list(subset_df["img"])

image_to_index = dict(zip(np.array(data['img']),data.index))
image_to_cls_number = dict(zip(np.array(data['img']), data['classname'].apply(lambda x:int(x[-1]))))
image_to_driver = dict(zip(np.array(data['img']), data['subject']))
image_to_cls = dict(zip(np.array(data['img']), data['classname']))

def image_to_tensor(image_path):
    """
    Function to convert  image to a 4-D tensor to supply as input for the keras model (tensorflow backend)
    Input : Path of the image
    Output : 4-D tensor
    """
    img = image.load_img(image_path, grayscale=True,target_size=(64, 64))
    image_array = image.img_to_array(img)
    image_array = image_array.astype('float32')/255
    return np.expand_dims(image_array, axis = 0) # covert the 3D tensor to 4D tensor of shape (1, 224, 224, 3)

def paths_to_tensor(images_path):
    """ Returns a stack of image tensors
    Input : Directory containg the images.
    Output : stack of tensors
    """
    list_of_tensors = [image_to_tensor(img_path) for img_path in tqdm(images_path)]
    return np.vstack(list_of_tensors)

img_tensors = paths_to_tensor(np.array(data["file_path"]))

def get_data(drivers_list):
    '''
    Input : List of driver labels
    Output : Returns the image tensors associated with that particular drivers
    '''
    input_data = []
    target_class_numbers = []
    target_class = []
    for each_driver in drivers_list:
        imgs_list = driver_to_img[each_driver]
        for each_image in imgs_list:
            input_data.append(img_tensors[image_to_index[each_image]])
            target_class_numbers.append(image_to_cls_number[each_image])
            target_class.append(image_to_cls[each_image])
    targets_OH = np.eye(len(np.unique(target_class_numbers)))[target_class_numbers]
    return np.array(input_data), np.array(targets_OH), np.array(target_class)

def architecture():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu',kernel_initializer = 'he_normal',padding='same',input_shape = img_tensors.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))

    return model

unique_drivers = np.unique(data.subject)
kf = KFold(n_splits = 5, random_state = 2017, shuffle=True)
fold = 0
epochs = 30
opt = Adam(lr=1e-3)
batch_size = 32
loss_dict = dict()

for train_index, test_index in kf.split(unique_drivers):
    fold += 1
    print("Running {}/{} fold".format(fold, kf.get_n_splits()))
    train_drivers = train_drivers = [unique_drivers[i] for i in train_index]
    test_drivers = [unique_drivers[i] for i in test_index]
    valid_drivers = random.sample(test_drivers, 3)
    test_drivers = [driver for driver in test_drivers if driver not in valid_drivers]
    train_data, train_targets, train_targets_names = get_data(train_drivers)
    print("Images in train data: ",len(train_data))
    valid_data, valid_targets, valid_targets_names = get_data(valid_drivers)
    print("Images in valid data: ",len(valid_data))
    test_data, test_targets, test_targets_names = get_data(test_drivers)
    print("Images in test data: ",len(test_data))

    model = architecture()
    model.compile(optimizer = opt, loss='categorical_crossentropy')
    weights_file = 'arch_1_cv/weights_fold_'+str(fold)+'_'+'adam'+'_arch_1_cv.hdf5'
    callbacks = [ModelCheckpoint(filepath=weights_file, verbose = 1, save_best_only = True),
                 EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, verbose=0, mode='auto')]
    print("Training the model")
    start = time.time()
    model.fit(train_data, train_targets,  validation_data=(valid_data, valid_targets),
          epochs = epochs, batch_size = batch_size, callbacks=callbacks, verbose = 1)
    print("Completed training in {} seconds".format(time.time() - start))

    model.load_weights(weights_file)

    print("Evaluationg model on test data")
    l_loss = model.evaluate(test_data, test_targets)
    print("Log loss on test data : {}".format(l_loss))
    loss_dict['fold'+str(fold)] = l_loss

least_loss_fold = min(loss_dict, key=loss_dict.get)
print("Least loss of {} in fold {} ".format(loss_dict[least_loss_fold], least_loss_fold))
print("Mean loss on test set : {} ".format(np.mean(list(loss_dict.values()))))
