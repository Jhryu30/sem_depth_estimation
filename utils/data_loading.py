import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import os


class Sem_Depth_Dataset(Dataset):
    # def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
    #     self.images_dir = Path(images_dir)
    #     self.masks_dir = Path(masks_dir)
    #     assert 0 < scale <= 1, 'Scale must be between 0 and 1'
    #     self.scale = scale
    #     self.mask_suffix = mask_suffix

    #     self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
    #     if not self.ids:
    #         raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
    #     logging.info(f'Creating dataset with {len(self.ids)} examples')

    # def __len__(self):
    #     return len(self.ids)
    ##############################################################
    def __init__(self, mode:str, transform=None):
        self.data_path = f"../Data/{mode}/"
        self.sem_list = os.listdir(os.path.join(self.data_path,'SEM/'))
        self.idx_sem_map = {i:d for i, d in enumerate(set(self.sem_list))}
        self.scale = 1.0
        self.mode = mode
        #self.data_filename = self.data_dict(self.sem_list)
        #self.transform = transform


    def __len__(self):
        # complete
        return len(self.sem_list)

    def __getitem__(self, idx):

        #convert_tensor = transforms.ToTensor()
        img_in_idx = self.idx_sem_map[idx]


        img_in_name = img_in_idx
        #img_in = np.array(Image.open(os.path.join(self.data_path+'SEM/',img_in_name)))
        img_in = Image.open(os.path.join(self.data_path+'SEM/',img_in_name))
        img_in = self.preprocess(img_in, self.scale, is_mask=False)

        if self.mode=="Test":
            return {'image': torch.as_tensor(img_in.copy()).float().contiguous(),
                    'f_name' : self.data_path+'SEM/'+img_in_name}

        else:
            #img_out = np.array(Image.open(os.path.join(self.data_path+'Depth/', img_in_name[:-9]+'.png')))
            img_out = Image.open(os.path.join(self.data_path+'Depth/', img_in_name[:-9]+'.png'))
            img_out = self.preprocess(img_out, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img_in.copy()).float().contiguous(),
            'mask': torch.as_tensor(img_out.copy()).long().contiguous()
        }

        #return img_in, img_out

    ##################################################################

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    # @staticmethod
    # def load(filename):
    #     ext = splitext(filename)[1]
    #     if ext in ['.npz', '.npy']:
    #         return Image.fromarray(np.load(filename))
    #     elif ext in ['.pt', '.pth']:
    #         return Image.fromarray(torch.load(filename).numpy())
    #     else:
    #         return Image.open(filename)

    # def __getitem__(self, idx):
    #     name = self.ids[idx]
    #     mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
    #     img_file = list(self.images_dir.glob(name + '.*'))

    #     assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
    #     assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
    #     mask = self.load(mask_file[0])
    #     img = self.load(img_file[0])

    #     assert img.size == mask.size, \
    #         f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

    #     img = self.preprocess(img, self.scale, is_mask=False)
    #     mask = self.preprocess(mask, self.scale, is_mask=True)

    #     return {
    #         'image': torch.as_tensor(img.copy()).float().contiguous(),
    #         'mask': torch.as_tensor(mask.copy()).long().contiguous()
    #     }


# class CarvanaDataset(BasicDataset):
#     def __init__(self, images_dir, masks_dir, scale=1):
#         super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
