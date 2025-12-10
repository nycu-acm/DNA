import random
from utils.util import normalize_xys
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from utils.util import corrds2xys
import codecs
import glob
import cv2
from PIL import ImageDraw, Image
transform_data = transforms.Compose([
    # transforms.Resize((113, 113)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5), std = (0.5))
])

script={"CHINESE":['CASIA_CHINESE', 'Chinese_content.pkl'],
        'JAPANESE':['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
        "ENGLISH":['CASIA_ENGLISH', 'English_content.pkl']
        }

"""
 loading generated online characters for evaluating the generation quality
"""
class Online_Dataset(Dataset):
    def __init__(self, data_path):
        lmdb_path = os.path.join(data_path, 'test')
        print("loading characters from", lmdb_path)
        if not os.path.exists(lmdb_path):
            raise IOError("input the correct lmdb path")

        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.writer_dict = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            self.indexes = list(range(0, self.num_sample))

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            character_id, coords, writer_id, coords_gt = data['character_id'], \
                data['coordinates'], data['writer_id'], data['coords_gt']
        try:
            coords, coords_gt = corrds2xys(coords), corrds2xys(coords_gt)
        except:
            print('Error in character format conversion')
            return self[index+1]
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id]),
                'coords_gt': torch.Tensor(coords_gt)}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['coords'].shape[0] for s in batch_data])
        max_len_gt = max([h['coords_gt'].shape[0] for h in batch_data])
        output = {'coords': torch.zeros((bs, max_len, 5)),  # preds -> (x,y,state) 
                  'coords_gt':torch.zeros((bs, max_len_gt, 5)), # gt -> (x,y,state) 
                  'coords_len': torch.zeros((bs, )),
                  'len_gt': torch.zeros((bs, )),
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,))}

        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords']
            h =  batch_data[i]['coords_gt'].shape[0]
            output['coords_gt'][i, :h] = batch_data[i]['coords_gt']
            output['coords_len'][i], output['len_gt'][i] = s, h
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']
        return output
    

class UserDataset(Dataset):
    def __init__(self, root='data', dataset='CHINESE', style_path='style_samples'):
        data_path = os.path.join(root, script[dataset][0])
        self.content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb')) #content samples
        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.style_path = glob.glob(style_path+'/*.[jp][pn]g')

    def __len__(self):
        return len(self.char_dict)
    
    def __getitem__(self, index):
        char = self.char_dict[index] # content samples
        char_img = self.content[char] 
        char_img = char_img/255. # Normalize pixel values between 0.0 and 1.0
        img_list = []
        for idx in range(len(self.style_path)):
            style_img = cv2.imread(self.style_path[idx], flags=0)
            style_img = cv2.resize(style_img, (64, 64))
            style_img = style_img/255.
            img_list.append(style_img)
        img_list = np.expand_dims(np.array(img_list), 1)
        
        return {'char_img': torch.Tensor(char_img).unsqueeze(0),
                'img_list': torch.Tensor(img_list),
                'char': char}

"""
 loading generated offline characters for calculating the Style Score
 takes 15 characters belonging to the same person as one input set
"""
class test_offline_Style_Dataset(Dataset):
    def __init__(self, root=None, is_train=True, num_img=15):
        self.is_train = is_train
        self.train_path = {}
        self.test_path = {}
        self.all_len = 0
        self.num_img = num_img
        if os.path.exists(os.path.join(root, 'writer_dict.pkl')):
            self.writer_dict = pickle.load(open(os.path.join(root, 'writer_dict.pkl'), 'rb'))
        else:
            self.writer_dict = [i for i in range(60)]
        all_jpg = glob.glob(os.path.join(root,'test2/*.jpg'))
        all_jpg = sorted(all_jpg)
        self.all_len = len(all_jpg)
        for path in all_jpg:
            pot = os.path.basename(path).split('_')[0]
            if pot in self.test_path:
                self.test_path[pot].append(path)
            else:
                self.test_path[pot] = []

        if self.is_train:
            data_path = self.train_path
        else:
            data_path = self.test_path
        self.indexs = data_path
        assert len(self.indexs) > 0, "input valid dataset!"
        print("loading %d datasets" % (len(self.indexs)))
        self.num_class = len(self.writer_dict)

    def __getitem__(self, index):
        num_random = self.num_img
        img_list = []
        label_list = []
        pot_name = random.choice(list(self.indexs.keys()))
        tmp_path = self.indexs[pot_name]
        random_indexs = random.sample(tmp_path,num_random)
        for path in random_indexs:
            img = Image.open(path).convert('L')
            data = transform_data(img)
            img_list.append(data)
        label = int(pot_name)
        img_list = torch.cat(img_list,0)
        if num_random==1:
            char = os.path.basename(path).split('_')[1]
            return img_list, label, char
        return img_list, label

    def __len__(self):
        return self.all_len

"""
 loading generated online characters for calculating the Content Score
"""
class Online_Gen_Dataset(Dataset):
    def __init__(self, data_path='lmdb', is_train=True):
        self.is_train = is_train
        if is_train:
            lmdb_path = os.path.join(data_path, 'train')
        else:
            lmdb_path = os.path.join(data_path, 'test')
        if not os.path.exists(lmdb_path):
            print("input the correct lmdb path")
            raise NotImplementedError
        
        self.char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
        self.writer_dict = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self.max_len = -1 
        self.alphabet = '' 
        self.cat_xy_grid = True

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            if len(self.alphabet) <= 0:
                self.indexes = list(range(0, self.num_sample))
            else:
                print('filter data out of alphabet')
                self.indexes = []
                for i in range(self.num_sample):
                    data_id = str(i).encode('utf-8')
                    data_byte = txn.get(data_id)
                    character_id = pickle.loads(data_byte)['character_id']
                    tag_char = self.char_dict[character_id]
                    if tag_char in self.alphabet:
                        self.indexes.append(i)

    def __getitem__(self, index):
        if self.is_train:
            index = index % (len(self))
        index = self.indexes[index]

        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            character_id, coords, writer_id = data['character_id'], data['coordinates'], data['writer_id']
        if self.is_train and self.max_len > 0:
            l_seq = sum([len(l)//2 for l in coords])
            if l_seq > self.max_len:
                print('skip {},{}'.format(index, self.char_dict[character_id]))
                return self[index+1] 
        try:
            coords = corrds2xys(coords) 
        except:
            print('error')
            return self[index+1]

        if coords is None:
            return self[index+1]
        else:
            pass
        return {'coords': torch.Tensor(coords),
                'character_id': torch.Tensor([character_id]),
                'writer_id': torch.Tensor([writer_id])}

    def __len__(self):
        return len(self.indexes)

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['coords'].shape[0] for s in batch_data])
        output = {'coords': torch.zeros((bs, max_len, 5)),
                  'coords_len': torch.zeros((bs, )),
                  'character_id': torch.zeros((bs,)),
                  'writer_id': torch.zeros((bs,))}
       
        for i in range(bs):
            s = batch_data[i]['coords'].shape[0]
            output['coords'][i, :s] = batch_data[i]['coords']
            output['coords_len'][i] = s
            output['character_id'][i] = batch_data[i]['character_id']
            output['writer_id'][i] = batch_data[i]['writer_id']

        return output