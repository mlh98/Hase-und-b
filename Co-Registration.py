import numpy as np
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import os


def load_nifti(file):
    data = nib.load(file)
    data = data.get_fdata()
    data = np.rot90(data, 3, axes=(0, 1))
    data = np.flip(data, axis=1)

    return data


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Load nifti image (you have to do it for each nifti file separately)
fixed_image = ants.image_read(os.path.join(script_dir, 'Messungen und Segmentierung', '01', 'data_original', '45_b1.nii'))
moving_image = ants.image_read(os.path.join(script_dir, 'Messungen und Segmentierung', '01', 'data_original', '45_b1.nii'))
nifti_img = nib.load(os.path.join(script_dir, 'Messungen und Segmentierung', '01', 'data_original', '45_b1.nii'))


# Number of b-values
n_b = np.shape(moving_image)[3]

fi = fixed_image[:,:,:,0]
fi = ants.from_numpy(fi)

processed_img = np.empty((moving_image.shape[0], moving_image.shape[1], moving_image.shape[2], n_b))
for b in range(n_b):
    print(b)
    mi = moving_image[:,:,:,b]
    mi = ants.from_numpy(mi)
    
    try:
        init_tx = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
        mytx = ants.registration(fixed=fi, moving=mi, initial_transform=init_tx['fwdtransforms'][0], type_of_transform='Affine')
        mywarpedimage = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'])
        mywarpedimage = mywarpedimage.numpy()
        processed_img[:,:,:,b] = mywarpedimage
    except:
        print('Registration was not successful. Keep the original image.')
        processed_img[:,:,:,b] = moving_image[:,:,:,b]

processed_nifti_img  = nib.Nifti1Image(processed_img, affine=nifti_img.affine, header=nifti_img.header)

nib.save(processed_nifti_img, os.path.join(script_dir, 'Messungen und Segmentierung', '01', 'data', 'output.nii'))