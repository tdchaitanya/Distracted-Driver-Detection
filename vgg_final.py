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
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D,GlobalMaxPooling2D, BatchNormalization ,Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input
from keras.optimizers import Adam, SGD

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

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def pretrained_path_to_tensor(img_path):
    # convert to Keras-friendly, 4D tensor
    img = path_to_tensor(img_path)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(img)


def pretrained_paths_to_tensor(img_paths):
    list_of_tensors = [pretrained_path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

img_tensors = pretrained_paths_to_tensor(np.array(data["file_path"]))

def get_data(drivers_list):
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

    model = VGG16(weights='imagenet', include_top=True)

    model.layers.pop()

    x = model.layers[-1].output
    predictions = Dense(10, activation='softmax')(x)

    model_transfer  = Model(inputs=model.input, outputs= predictions)

    for layer in model.layers[:7]:
        layer.trainable = False
    for layer in model.layers[7:20]:
        layer.trainable = True
    for layer in model.layers[20:22]:
        layer.trainable = True
    return model_transfer

unique_drivers = data.subject.unique()
kf = KFold(n_splits = 5, random_state = 2017, shuffle=False)
fold = 0
epochs = 25
# opt = Adam(lr=1e-3)
opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9)
batch_size = 16
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
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    weights_file = 'vgg_model_weights/weights_fold_'+str(fold)+'_'+'sgd_vgg.hdf5'
    callbacks = [ModelCheckpoint(filepath=weights_file, verbose = 1, save_best_only = True),
                 EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, mode='auto')]
    print("Training the model")
    start = time.time()
    model.fit(train_data, train_targets,  validation_data=(valid_data, valid_targets),
          epochs = epochs, batch_size = batch_size, callbacks=callbacks,verbose = 1)
    print("Completed training in {} minutes".format((time.time() - start)/60))

    model.load_weights(weights_file)

    print("Evaluationg model on test data")
    l_loss = model.evaluate(test_data, test_targets)
    print("Log loss on test data : {}".format(l_loss))
    loss_dict['fold'+str(fold)] = l_loss

least_loss_fold = min(loss_dict, key=loss_dict.get)
print("Least loss of {} in fold {} ".format(loss_dict[least_loss_fold], least_loss_fold))
print("Mean loss on test set : {} ".format(np.mean(list(loss_dict.values()))))
