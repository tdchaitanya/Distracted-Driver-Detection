
import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Activation,MaxPooling2D, GlobalMaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from keras.optimizers import Adam
import tensorflow as tf
import gc

if tf.test.gpu_device_name():
    print("GPU device found : {} ".format(tf.test.gpu_device_name()))
else :
    print("No GPU device found, CPU will be used")



########################################################################################

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



# Just replace the architecture being used
predictions_file_name = 'predictions_arch1_cv_loss_2.0_corr.csv'

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu',kernel_initializer = 'he_normal',padding='same',input_shape = (64, 64, 1)))
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

epochs = 30
opt = Adam(lr=1e-3)
batch_size = 32

# compiling the model
model.compile(optimizer = opt, loss='categorical_crossentropy')
weights_file = 'arch_1_cv/weights_fold_1_adam_arch_1_cv.hdf5'
callbacks = [ModelCheckpoint(filepath=weights_file, verbose = 1, save_best_only = True),
             EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, verbose=0, mode='auto')]

model.load_weights(weights_file)
############################################################################################

files_test  = os.listdir('../imgs/test/')
files_test_path = [os.path.join('../imgs','test', i) for i in files_test]

print("Reading first half of test images")
test_set_images_1 = paths_to_tensor(files_test_path[:len(files_test_path)//2])


predictions_test_1 = [model.predict(np.expand_dims(tensor,axis = 0), batch_size=64) for tensor in tqdm(test_set_images_1)]

data = pd.DataFrame(columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])

data['img'] = files_test[:(len(files_test_path)//2)]

predictions_test = np.array(predictions_test_1)
preds_arrs = predictions_test.reshape(len(files_test_path)//2, 10)

data[['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']] = preds_arrs

data.to_csv('predictions_set1_arch_1.csv')

print("First set predictions and writing the results completed")
del test_set_images_1
del predictions_test_1
del predictions_test
del preds_arrs
del data
gc.collect()

print("Reading second half of test images")
test_set_images_2 = paths_to_tensor(files_test_path[len(files_test_path)//2:])
predictions_test_2 = [model.predict(np.expand_dims(tensor,axis = 0), batch_size=64) for tensor in tqdm(test_set_images_2)]

data = pd.DataFrame(columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])

data['img'] = files_test[(len(files_test_path)//2):]

predictions_test = np.array(predictions_test_2)
preds_arrs = predictions_test.reshape(len(files_test_path)//2, 10)

data[['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']] = preds_arrs

data.to_csv('predictions_set2_arch_1.csv')
print("Second set predictions and writing the results completed")

del test_set_images_2
del predictions_test_2
del predictions_test
del preds_arrs
del data
gc.collect()

print("Reading two predictions files")
set1 = pd.read_csv('predictions_set1_arch_1.csv')
set2 = pd.read_csv('predictions_set2_arch_1.csv')

final_predictions = pd.concat([set1, set2])
final_predictions.to_csv(predictions_file_name)
print("Predictions in {} number of rows in dataframe {}".format(predictions_file_name, len(final_predictions)))
