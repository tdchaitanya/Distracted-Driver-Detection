import numpy as np
import os
import time
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import tensorflow as tf

# total of 10 categories and 22424 training images
driver_images = load_files('../imgs/train/') # load the files and categores using sklearn load_files function
driver_filenames = np.array(driver_images["filenames"]) # array of filenames
driver_targets = np.array(driver_images["target"]) # targets in form of integers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
driver_target_names = driver_images["target_names"] # name of the targets ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

# safeside --> remove files which are not images.
bool_to_keep = [filename.endswith('jpg') for filename in driver_filenames]

# keep only files with jpg extension
driver_filenames = driver_filenames[bool_to_keep]
driver_targets = driver_targets[bool_to_keep]

if tf.test.gpu_device_name():
    print("GPU device found : {} ".format(tf.test.gpu_device_name()))
else :
    print("No GPU device found, CPU will be used")

def image_to_tensor(image_path):
    """
    Function to convert  image to a 4-D tensor to supply as input for the keras model (tensorflow backend)
    Input : Path of the image
    Output : 4-D tensor
    """
    img = image.load_img(image_path, target_size=(224, 224))
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

# split the training data into train and test sets, 80% into train data and 20% into test data
filenames_train, filenames_test, targets_train, targets_test = train_test_split(driver_filenames,
                                                                                list(driver_targets),
                                                                                test_size=.2,
                                                                                stratify=list(driver_targets),
                                                                                random_state=2017)
# split the training data into validation and test sets, 10% as validation data and 10% as test data
filenames_valid, filenames_test, targets_valid, targets_test = train_test_split(filenames_test,
                                                                                            targets_test,
                                                                                            test_size=.5,
                                                                                            stratify=list(targets_test),
                                                                                            random_state=2017)
# forming the stack of image tensors for train, test and validation data
print("Converting image to tensor type objects")
images_train = paths_to_tensor(filenames_train)
images_test = paths_to_tensor(filenames_test)
images_valid = paths_to_tensor(filenames_valid)

# one hot encoding the target values
train_targets_oh = np.eye(len(np.unique(targets_train)))[targets_train]
valid_targets_oh = np.eye(len(np.unique(targets_valid)))[targets_valid]
test_targets_oh = np.eye(len(np.unique(targets_test)))[targets_test]

# architecture of the CNN
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


# compiling the model
model.compile(optimizer = 'adam', loss='categorical_crossentropy')
epochs = 30
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto'),
             ModelCheckpoint(filepath='arch_1_weights/weights_ep_'+str(epochs)+'_adam_arch_1.hdf5', verbose = 1, save_best_only = True)]

print("Training the model : ")
start = time.time()
# fitting the model on training data
model.fit(images_train, train_targets_oh,  validation_data=(images_valid, valid_targets_oh),
         epochs = epochs, batch_size = 1, callbacks=callbacks, verbose = 1)
print("Completed training in {} minutes".format((time.time() - start)/60))

model.load_weights('arch_1_weights/weights_ep_'+str(epochs)+'_adam_arch_1.hdf5')

print("Making predictions on test data")
predictions = np.array([model.predict(np.expand_dims(tensor, axis = 0), batch_size = 64) for tensor in tqdm(images_test)])
predictions = predictions.reshape(images_test.shape[0], 10)

# get correct classes for images in test data
targets_test_names = targets_test[:]
targets_test_names = ['c' + str(i) for i in targets_test_names]
l_loss = log_loss(targets_test_names, predictions)
print("Log loss on test data : {}".format(l_loss) )
