"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir,file_list,opts,state):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if state == 'train' or opts.checkpoint_path !=None:
                    images.append(path)
    if state=='train' and file_list != None and opts.augmentation:
        file_list=file_list.item()
        for key in file_list:
            count=int(27000/len(file_list[key]))
            for file in file_list[key]:
                path=os.path.join(root,file+fname[-4:])
                for i in range(count):
                    images.append(path)
    if state=='test' and file_list != None and opts.augmentation:
        file_list=file_list.item()
        for key in file_list:
            for file in file_list[key]:
                path=os.path.join(root,file+fname[-4:])
                images.append(path)

    return images
