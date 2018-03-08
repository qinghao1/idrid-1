import numpy as np, keras.callbacks as keras, sys
from unet.unet import Unet

data_dir = 'images/data/'
lesion_types = ['MA','HE','EX','SE','OD']
selected_type = 'MA' # Default
if sys.argv and sys.argv[1] in lesion_types:
	selected_type = sys.argv[1]

# Get U-Net model
cnn = Unet(512, 512, 3).model
print('-' * 30)
print("Initialized U-Net")
print('-' * 30)

# Get data
train_data = np.load(data_dir + 'train_data.npy')
train_data = train_data.astype('float32')
train_data /= 255 # Scale to 0..1
print("Loaded training data")

train_labels = np.load(data_dir + 'train_label_' + selected_type + '.npy')
train_labels = train_labels.astype('float32')
print("Loaded training label for type " + selected_type)

print("Training dataset size: %d" % len(train_data))
print('-' * 30)

# Train U-Net

print('Loading pre-trained weights...')
cnn.load_weights('models/MA_unet_10.hdf5')
print('Loaded weights')

model_checkpoint = keras.ModelCheckpoint('models/' + selected_type + '_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')

cnn.fit(train_data, train_labels, batch_size=1, epochs=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

print('-' * 30)
print('Model successfully trained!')