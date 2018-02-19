import scipy.io as sio, numpy as np, glob, imageio, re, copy, random, sklearn.model_selection as sklearn

images_dir = '../images/raw/*.mat'
save_dir = '../images/jpg/'

image_files = glob.glob(images_dir)
image_files.sort()

original_image_label = 'I_cropped'
ground_truth_label = 'GT'
lesion_types = ['MA','HE','EX','SE','OD']

file_name_regex = '([^/]*)$'

# Data arrays to be saved as .npy files for model training and testing
all_data = []
all_labels = {}
training_data = []
testing_data = []
training_labels = {}
testing_labels = {}

for lesion in lesion_types:
	all_labels[lesion] = []
	training_labels[lesion] = []
	testing_labels[lesion] = []

for idx, filename in enumerate(image_files):
	if idx > 10: continue
	stripped_filename = re.search(file_name_regex, filename).group(1)

	# Load data
	data = sio.loadmat(filename)
	image = data[original_image_label]
	ground_truth = data[ground_truth_label]

	# Save original image as JPG
	# imageio.imwrite(
	# 	save_dir
	# 	+ stripped_filename[:-4]
	# 	+ '.jpg',
	# 	image)

	# Save original image to all_data
	all_data.append(image)

	# Generate and save ground truth images
	if ground_truth.shape[0]:
		# Use ground truth masks to show the parts we want
		for lesion in lesion_types:
			gt_label = lesion + '_mask' # 'MA_mask'
			gt_mask = ground_truth[gt_label][0][0] # Array of 1s and 0s
			if gt_mask.shape[0]: # Check if gt_mask is empty 2d list [[]]

				# Add ground truth mask to all_labels
				all_labels[lesion].append(gt_mask)

				# # Save ground truth as jpg
				# gt_image = filter(image, gt_mask)
				# imageio.imwrite(
				# 	save_dir
				# 	+ stripped_filename[:-4]
				# 	+ '_'
				# 	+ lesion
				# 	+ '.jpg',
				# 	gt_image)

			else:
				all_labels[lesion].append(np.zeros(image.shape[:-1] + (1,)))

	else:
		for lesion in lesion_types:
			all_labels[lesion].append(np.zeros(image.shape[:-1] + (1,)))


	print('Processed file %d of %d: %s' 
		% (idx + 1, len(glob.glob(images_dir)), stripped_filename))

all_data = np.array(all_data)
for lesion in lesion_types:
	all_labels[lesion] = np.array(np.reshape(all_labels[lesion], all_data[0].shape[:-1]))

# Train-test split
training_data, testing_data,
training_labels['MA'], testing_labels['MA'], 
training_labels['HE'], testing_labels['HE'], 
training_labels['EX'], testing_labels['EX'], 
training_labels['SE'], testing_labels['SE'], 
training_labels['OD'], testing_labels['OD'] = sklearn.train_test_split(
												all_data,
												all_labels['MA'],
												all_labels['HE'],
												all_labels['EX'],
												all_labels['SE'],
												all_labels['OD'],
												test_size = 0.1)


def filter(img, mask):
	img2 = copy.deepcopy(img)
	num_channels = img.shape[2]
	for idx1, mat in enumerate(img):
		for idx2, arr in enumerate(mat):
			if(mask[idx1][idx2] == 0):
				img2[idx1][idx2] = (0,) * num_channels
				return img2