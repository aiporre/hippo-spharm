# checks nii.gz files in sub-*/ses-*/anat/ folder orientation in the input dataset
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Check orientation of nifti files in the dataset')
parser.add_argument('dataset_path', help='Path to the dataset')
parser.add_argument('-b', '--brain', action='store_true', help='Use brain extracted images')

args = parser.parse_args()
dataset_path = args.dataset_path
brain_extraction = args.brain
subs = [f for f in os.listdir(dataset_path) if f.startswith('sub')]
if len(subs) == 0:
    print(f"Directory {dataset_path} has not sub-x directories containing data")
    print("Done")
    sys.exit(0)
# find the inputs as dict
def get_mri(sub):
    files = os.listdir(os.path.join(dataset_path,sub,'anat'))
    if brain_extraction:
        mri_file = [f for f in files if f.endswith('_brain.nii.gz')][0]
    else:
        mri_file = [f for f in files if f.endswith('_corrected.nii.gz')][0]
    return os.path.join(dataset_path,sub, 'anat', mri_file)
print('one mri file' , get_mri(subs[0]))
# make a list of inputs
files_input = [get_mri(sub) for sub in subs]
# make a list of outputs changing a suffix -corrected.nii.gz
def make_output(f_input, sub,  suffix):
    var_path = os.path.dirname(f_input)
    print('subs', sub)
    var_path = os.path.join(var_path,  sub + '_' + suffix + '.nii.gz')
    return var_path
if brain_extraction:
    files_corrected = [make_output(f_in, sub, 'brain') for f_in, sub in zip(files_input, subs)]
else:
    files_corrected = [make_output(f_in, sub, 'corrected') for f_in, sub in zip(files_input, subs)]
print('first file input,', files_input[0])
print('first file corrected,', files_corrected[0])
# open file and check orientation with the affine flag
import nibabel as nib


def apply_affine(data, affine):
    new_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                coord = (i, j, k)
                transformed_coord = nib.affines.apply_affine(affine, coord)
                x, y, z = transformed_coord
                new_data.append(data[x, y, z])

    new_data = np.asarray(new_data).reshape(data.shape)
    return new_data

def check_orientation(sub, f):
    img = nib.load(f)
    # print('header before', img.header)
    data = img.get_fdata()
    print('current orientation', nib.aff2axcodes(img.affine))
    t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    x, y, z = nib.aff2axcodes(img.affine)

    def swap(a, b):
        return b, a

    sets = [('L', 'R'), ('P', 'A'), ('I', 'S')]
    def assign_set(x):
        for s in sets:
            if x in s:
                return sets.index(s)
    x_num = assign_set(x)
    y_num = assign_set(y)
    z_num = assign_set(z)
    print('x, y, z = ', x, y, z)
    print('x_num, y_num, z_num = ', x_num, y_num, z_num)
    # swap axes
    if y_num == 0:
        # swap columns 1 and x_num
        t[0,0], t[0,1] = t[0,1], t[0,0]
        t[1,0], t[1,1] = t[1,1], t[1,0]
        t[2,0], t[2,1] = t[2,1], t[2,0]
        x_num, y_num = swap(x_num, y_num)
        x, y = swap(x, y)
    if z_num == 0:
        t[0, 0], t[0, 2] = t[0, 2], t[0, 0]
        t[1, 0], t[1, 2] = t[1, 2], t[1, 0]
        t[2, 0], t[2, 2] = t[2, 2], t[2, 0]
        x_num, z_num = swap(x_num, z_num)
        x, z = swap(x, z)
    if z_num == 1:
        t[0, 1], t[0, y_num] = t[0, y_num], t[0, 1]
        t[1, 1], t[1, y_num] = t[1, y_num], t[1, 1]
        t[2, 1], t[2, y_num] = t[2, y_num], t[2, 1]
        y_num, z_num = swap(y_num, z_num)
        y, z = swap(y, z)
    if x != 'L':
        # columns 0 multiplied by -1
        t[:, 0] = t[:, 0] * -1
        x = 'L'
    if y != 'P':
        t[:, 1] = t[:, 1] * -1
        y = 'P'
    if z != 'I':
        t[:, 2] = t[:, 2] * -1
        z = 'I'
    print('np.dot(img.affine, t) = ', img.affine, " * ", t, " = ", np.dot(img.affine, t))
    new_affine = np.dot(img.affine, t)
    print(t)
    print('x, y, z = ', x, y, z)
    print('x_num, y_num, z_num = ', x_num, y_num, z_num)

    a, b, c = nib.aff2axcodes(img.affine)
    data_orig = data
    ort = nib.orientations.axcodes2ornt((a, b, c))
    print('ort = ', ort)
    data = nib.orientations.apply_orientation(data, ort)
    # data = apply_affine(data, t)
    # plot 3 slices to check
    # fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    # ax[0,0].imshow(data[:, :, data.shape[2] // 2])
    # ax[0,1].imshow(data[:, data.shape[1] // 2, :])
    # ax[0,2].imshow(data[data.shape[0] // 2, :, :])
    # ax[1,0].imshow(data_orig[:, :, data_orig.shape[2] // 2])
    # ax[1,1].imshow(data_orig[:, data_orig.shape[1] // 2, :])
    # ax[1,2].imshow(data_orig[data_orig.shape[0] // 2, :, :])
    # # make titles
    # ax[0,0].set_title('new axial')
    # ax[0,1].set_title('new coronal')
    # ax[0,2].set_title('new sagittal')
    # ax[1,0].set_title('org axial')
    # ax[1,1].set_title('org coronal')
    # ax[1,2].set_title('org sagittal')
    # plt.show()
    img = nib.Nifti1Image(data, new_affine)
    print('img.header', img.header)
    # save the image
    f_out = make_output(f, sub, 'reoriented')
    nib.save(img, f_out)
    x, y, z = nib.aff2axcodes(img.affine)
    print('new orientation', nib.aff2axcodes(img.affine))
    fixed_orientation = x != 'R' or y != 'P' or z != 'I'
    return fixed_orientation

# check orientation of inputs
for sub, f in zip(subs, files_input):
    if check_orientation(sub, f):
        print('Orientation of', f, 'was fixed')
    else:
        print('Orientation of', f, 'is correct')
