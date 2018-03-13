import scipy.io as sio, numpy as np, skimage.transform as imtransform, glob, imageio, re, copy

images_dir = 'images/raw/*.mat'
save_dir = 'images/jpg/'

image_files = glob.glob(images_dir)
image_files.sort()

# Resize options
resize_image = True
new_dimensions = (512, 512,)

original_image_label = 'I_cropped'
ground_truth_label = 'GT'
lesion_types = ['MA','HE','EX','SE','OD']

file_name_regex = '([^/]*)$'

def maskImage(img, mask):
	img2 = copy.deepcopy(img)
	num_channels = img.shape[2]
	for idx1, mat in enumerate(img):
		for idx2, arr in enumerate(mat):
			if(mask[idx1][idx2] == 0):
				img2[idx1][idx2].fill(0)
	return img2

for idx, filename in enumerate(image_files):
	stripped_filename = re.search(file_name_regex, filename).group(1)

	# Load data
	data = sio.loadmat(filename)
	image = data[original_image_label]
	ground_truth = data[ground_truth_label]

	if resize_image:
		image = imtransform.resize(image, new_dimensions)

	# Save original image as JPG
	imageio.imwrite(
		save_dir
		+ stripped_filename[:-4]
		+ '.jpg',
		image)

	# Generate and save ground truth images
	if ground_truth.shape[0]:
		# Use ground truth masks to show the parts we want
		for lesion in lesion_types:
			gt_label = lesion + '_mask' # 'MA_mask'
			gt_mask = ground_truth[gt_label][0][0] # Array of 1s and 0s
			if gt_mask.shape[0]: # Check if gt_mask is empty 2d list [[]]
				# Save ground truth as jpg
				if resize_image:
					gt_mask = imtransform.resize(gt_mask, new_dimensions)
					np.round(gt_mask) # Round decimals to 0 or 1
				gt_image = maskImage(image, gt_mask)
				imageio.imwrite(
					save_dir
					+ stripped_filename[:-4]
					+ '_'
					+ lesion
					+ '.jpg',
					gt_image)


	print('Processed file %d of %d: %s' 
		% (idx + 1, len(image_files), stripped_filename))