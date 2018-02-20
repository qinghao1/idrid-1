import numpy as np, keras.callbacks as keras, sys
from unet.unet import Unet

data_dir = 'images/data/'
results_dir = 'results/'
lesion_types = ['MA','HE','EX','SE','OD']
selected_type = 'MA' # Default
if sys.argv and sys.argv[0] in lesion_types:
	selected_type = sys.argv[0]

# Get U-Net model
cnn = Unet(1024, 1024, 3).model
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

print(train_labels.shape)

test_data = np.load(data_dir + 'test_data.npy')
test_data = test_data.astype('float32')
test_data /= 255 # Scale to 0..1
print("Loaded test data")

print("Training dataset size: %d, Test dataset size: %d" % (len(train_data), len(test_data)))
print('-' * 30)

# Train U-Net

model_checkpoint = keras.ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')

cnn.fit(train_data, train_labels, batch_size=4, epochs=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

print('-' * 30)

# Predict on test data

print('Predicting test data...')
test_predictions = cnn.predict(test_data, batch_size=1, verbose=1)
np.save(results_dir + 'predictions', test_predictions)