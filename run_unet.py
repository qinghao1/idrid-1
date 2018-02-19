import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *

from unet.unet import Unet

# Get U-Net model
cnn = Unet(1024, 1024, 3)

# Get data



print("loading data")
imgs_train, imgs_mask_train, imgs_test = self.load_data()
print("loading data done")
model = self.get_unet()
print("got unet")

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')
model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

print('predict test data')
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('../results/imgs_mask_test.npy', imgs_mask_test)