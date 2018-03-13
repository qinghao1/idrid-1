import scipy.io as sio, numpy as np, skimage.transform as imtransform, glob, imageio, re, copy, random

train_test_ratio = 0.9

# Resize options
resize_image = True
new_dimensions = (512, 512,)

images_dir = 'images/raw/*.mat'
save_dir = 'images/data/'

image_files = glob.glob(images_dir)
image_files.sort()
train_image_files = random.sample(image_files, int(0.9 * len(image_files)))
test_image_files = [f for f in image_files if f not in train_image_files]

original_image_label = 'I_cropped'
ground_truth_label = 'GT'
lesion_types = [
	'MA',
	'HE',
	'EX',
	'SE',
]

file_name_regex = '([^/]*)$'

# Lists of tests and training images and data
train_data = []
train_labels = {
	'MA':[], 
	'HE':[],
	'EX':[],
	'SE':[],
}
test_data = []
test_labels = {
	'MA':[], 
	'HE':[],
	'EX':[],
	'SE':[],
}

# Training files

for idx, filename in enumerate(train_image_files):
	stripped_filename = re.search(file_name_regex, filename).group(1)

	# Load data
	data = sio.loadmat(filename)
	image = data[original_image_label]
	ground_truth = data[ground_truth_label]

	if resize_image:
		resized_image = imtransform.resize(image, new_dimensions)
		train_data.append(resized_image)
	else:
		train_data.append(image)
	
	for lesion in lesion_types:
		gt_label = lesion + '_mask' # 'MA_mask'
		if ground_truth.shape[0] and ground_truth[gt_label][0][0].shape[0]:
			gt_mask = ground_truth[gt_label][0][0].reshape(image.shape[:-1] + (1,))
			if resize_image:
				gt_mask = imtransform.resize(gt_mask, new_dimensions)
				# Interpolation makes mask become decimal values, ceil them to 1
				gt_mask[gt_mask > 0] = 1
			train_labels[lesion].append(gt_mask)
		else:
			if resize_image:
				train_labels[lesion].append(np.zeros(new_dimensions + (1,)))
			else:
				train_labels[lesion].append(np.zeros(image.shape[:-1] + (1,)))


	print('Processed training file %d of %d: %s' 
		% (idx + 1, len(train_image_files), stripped_filename))

# Testing files

for idx, filename in enumerate(test_image_files):
	stripped_filename = re.search(file_name_regex, filename).group(1)

	# Load data
	data = sio.loadmat(filename)
	image = data[original_image_label]
	ground_truth = data[ground_truth_label]

	if resize_image:
		resized_image = imtransform.resize(image, new_dimensions)
		test_data.append(resized_image)
	else:
		test_data.append(image)

	for lesion in lesion_types:
		gt_label = lesion + '_mask' # 'MA_mask'
		if ground_truth.shape[0] and ground_truth[gt_label][0][0].shape[0]:
			gt_mask = ground_truth[gt_label][0][0].reshape(image.shape[:-1] + (1,))
			if resize_image:
				gt_mask = imtransform.resize(gt_mask, new_dimensions)
				# Interpolation makes mask become decimal values, ceil them to 1
				gt_mask[gt_mask > 0] = 1
			test_labels[lesion].append(gt_mask)
		else:
			if resize_image:
				test_labels[lesion].append(np.zeros(new_dimensions + (1,)))
			else:
				test_labels[lesion].append(np.zeros(image.shape[:-1] + (1,)))


	print('Processed training file %d of %d: %s' 
		% (idx + 1, len(test_image_files), stripped_filename))

# Save arrays to .npy files

np.save(save_dir + 'train_data', train_data)
print("Successfully saved train_data")
np.save(save_dir + 'test_data', test_data)
print("Successfully saved test_data")
for lesion in lesion_types:
	np.save(save_dir + 'train_label_' + lesion, train_labels[lesion])
	print("Successfully saved train_label_" + lesion)
	np.save(save_dir + 'test_label_' + lesion, test_labels[lesion])
	print("Successfully saved test_label_" + lesion)