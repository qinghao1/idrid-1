import numpy as np, imageio, copy

def maskImage(img, mask):
	img2 = copy.deepcopy(img)
	num_channels = img.shape[2]
	for idx1, mat in enumerate(img):
		for idx2, arr in enumerate(mat):
			if(mask[idx1][idx2] == 0):
				img2[idx1][idx2].fill(0)
	return img2


file_path = 'images/data/train_data.npy'
output_path = 'images/jpg/train_set/'
img_data = np.load(file_path)
print("Loaded image data.")

label_path = 'images/data/train_label_EX.npy'
label_data = np.load(label_path)
print('Loaded label data.')

for idx, img in enumerate(img_data):
	imageio.imwrite(output_path + str(idx) + '_orig.jpg', img)
	imageio.imwrite(output_path + str(idx) + '_EX.jpg', maskImage(img, label_data[idx]))