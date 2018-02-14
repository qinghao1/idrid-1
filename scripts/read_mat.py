import scipy.io as sio, glob, imageio, re

images_dir = '../images/raw/*.mat'
save_dir = '../images/jpg/'

original_image_label = 'I_cropped'
ground_truth_label = 'GT'
gt_MA_label = 'MA_mask'
gt_HE_label = 'HE_mask'
gt_EX_label = 'EX_mask'
gt_SE_label = 'SE_mask'
gt_OD_label = 'OD_mask'
gt_labels = [gt_MA_label, gt_HE_label, gt_EX_label, gt_SE_label, gt_OD_label]

file_name_regex = '([^/]*)$'

for idx, filename in enumerate(glob.glob(images_dir)):
	stripped_filename = re.search(file_name_regex, filename).group(1)

	# Load data
	data = sio.loadmat(filename)
	image = data[original_image_label]
	ground_truth = data[ground_truth_label]

	# Save original image
	imageio.imwrite(
		save_dir
		+ stripped_filename[:-4]
 		+ '.jpg',
 		image)

	# Save ground truths
	for gt_label in gt_labels:
		# print(ground_truth[gt_label][0][0])
		imageio.imwrite(
			save_dir
			+ stripped_filename[:-4]
			+ gt_label
			+ '.jpg',
			ground_truth[gt_label][0])


	print('Processed file %d of %d: %s' 
		% (idx + 1, len(glob.glob(images_dir)), stripped_filename))
