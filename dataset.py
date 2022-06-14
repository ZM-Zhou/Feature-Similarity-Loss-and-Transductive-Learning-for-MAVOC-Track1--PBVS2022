import os
from PIL import Image, ImageFile
import random

import torch
import torch.utils.data as data
from torchvision import transforms

seed = 310
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


ImageFile.LOAD_TRUNCATED_IMAGES = True
class EOSAR_Dataset(data.Dataset):
    NUM_CLS = 10
    def __init__(self,
                 data_path,
                 data_list_path,
                 image_transform=None,
                 read_EO=True,
                 image_to_RGB=False,
                 uniform_sample=False,
                 EO_transfom=None,
                 test_path=None):
        super().__init__()

        self.data_path = data_path
        self.image_transform = image_transform
        self.read_EO = read_EO
        self.image_to_RGB = image_to_RGB
        self.uniform_sample = uniform_sample
        self.EO_trans = EO_transfom
        self.test_path = test_path

        self.data_list = self._get_data_list(data_list_path)

        if self.uniform_sample:
            self.update_data_list()
        else:
            self.used_list = self.data_list

        if self.read_EO and self.EO_trans is None:
            self.EO_trans = \
            transforms.Compose([transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(10),
                                transforms.RandomCrop(224),
                                transforms.ColorJitter(0.2, 0.2, 0, 0),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
            
    def _get_data_list(self, data_list_path):
        with open(data_list_path, 'r') as f:
            lines = f.read().splitlines()

        self.data_per_class = {'0':[], '1':[], '2':[], '3':[], '4':[],
                               '5':[], '6':[], '7':[], '8':[], '9':[],
                               '-1':[]}
        for line in lines:
            str_label, _ = line.split(' ')
            self.data_per_class[str_label].append(line)

        return lines
    
    def __len__(self):
        return len(self.used_list)

    def __getitem__(self, f_idx):
        inputs = {}
        file_info = self.used_list[f_idx]
        str_label, data_num = file_info.split(' ')

        inputs['Idx'] = data_num

        # build label
        if str_label.startswith('_'):
            data_label = int(str_label[1:])
        else:
            data_label = int(str_label)
        data_label = torch.tensor(data_label)
        inputs['Label'] = data_label
   
        # read and process images
        if str_label.startswith('_'):
            SAR_path = os.path.join(self.test_path,'-1',
                                    'SAR_{}.png'.format(data_num))
        else:
            SAR_path = os.path.join(self.data_path, str_label,
                                    'SAR_{}.png'.format(data_num))
        SAR_image = self._read_image(SAR_path)
        SAR_image = self.image_transform(SAR_image)
        inputs['SAR'] = SAR_image
        if self.read_EO:
            EO_path = os.path.join(self.data_path, str_label,
                                  'EO_{}.png'.format(data_num))
            EO_image = self._read_image(EO_path)
            EO_image = self.EO_trans(EO_image)

            inputs['EO'] = EO_image
        
        return inputs
    
    def _read_image(self, path):
        img = Image.open(path).convert('L')
        if self.image_to_RGB:
            img = img.convert('RGB')
        return img

    def update_data_list(self, extra_list=None):
        if self.uniform_sample:
            new_train_list = []
            for class_num, class_list in self.data_per_class.items():
                if class_num == '-1':
                    continue
                select_list = random.sample(class_list, self.uniform_sample)
                new_train_list += select_list
            self.used_list = new_train_list
        else:
            pass
        
        if extra_list is not None:
            self.used_list += extra_list


        



       





