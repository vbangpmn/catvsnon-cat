Dataset is in .h5 file. It is a file format that could store the data - along with its meta-data - in the form of a hierarchy. 
Import h5py to interact with a dataset that is stored in an H5 file. It contains

train_catvnoncat.h5 - a training set of images labeled as cat (y=1) or non-cat (y=0)
test_catvnoncat.h5 - a test set of images labeled as cat or non-cat
Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px)
