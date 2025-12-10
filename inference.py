import lmdb
import pickle
import random
import numpy as np
import os
from utils.util import normalize_xys
import torch
import numpy as np
import cv2
import json
from copy import deepcopy

class DataProcessing:
    def __init__(self, style_path='./data/CHINESE/train_tra_all_img/Big5_001.pkl',
                content_img_path='./data/src/src.pkl',
                component_path='./data/CHINESE/dictionary_ids.txt',
                char_dict_path='./data/CHINESE/character_dict_t.pkl',
                save_path='./results_test/style'):

        os.makedirs(save_path, exist_ok=True)
        ##### Load file ####
        self.style_path = style_path

        ###/ component dictionary
        with open(component_path, 'r') as f:
            self.component = json.loads(f.readline().replace('\'', '\"'))

        ###/ content img
        self.content = pickle.load(open(content_img_path, 'rb')) #content samples

        self.char_dict = pickle.load(open(char_dict_path, 'rb'))
        ###/
        if self.style_path.endswith('pkl'):
            self.writer_id = self.style_path.split('/')[-1].split('.')[0]
        else:
            self.writer_id = os.path.basename(self.style_path)

        ##### Style Processing #####
        if self.style_path.endswith('pkl'):
            with open(self.style_path, 'rb') as f:
                style_samples = pickle.load(f)
        else:
            self.img_list = []
            for sf in os.listdir(self.style_path):
                tmp_img = cv2.imread(os.path.join(self.style_path, sf), cv2.IMREAD_GRAYSCALE)
                tmp_img = cv2.resize(tmp_img, (64, 64))
                _, tmp_img = cv2.threshold(tmp_img, 190, 255, cv2.THRESH_BINARY)
                tmp_img = tmp_img.astype(np.float32)
                tmp_img = tmp_img/255.
                self.img_list.append(tmp_img)

        ##### Save writer's style image #####
        os.makedirs(os.path.join(save_path, 'style'), exist_ok=True)
        idx_img=0
        new_img = np.zeros((64*4, 64*4))
        for row in range(4):
            for col in range(4):
                new_img[col*64:(col+1)*64, row*64:(row+1)*64] = self.img_list[idx_img]*255

                idx_img += 1
                if idx_img >= len(self.img_list):
                    break
            if idx_img >= len(self.img_list):
                break

        cv2.imwrite(os.path.join(save_path, 'style', f'{self.writer_id}_writer_style.png'), new_img)

        ##### Fixed #####
        self.img_list = np.expand_dims(np.array(self.img_list), 1) # [N, C, H, W], C=1
        self.img_list = np.expand_dims(np.array(self.img_list), 0) # [N, C, H, W], C=1

    def __call__(self, tag_char='罢'):
        ##### char_img #####
        char_com = self.component['valid'][tag_char]['components']
        char_struct = [self.component['valid'][tag_char]['struct']]

        ###/ content img
        char_img = self.content[tag_char] # content samples
        if isinstance(char_img, list):
            char_img = np.array(char_img)
        char_img = char_img/255. # Normalize pixel values between 0.0 and 1.0

        ##### Fixed #####
        char_com = np.expand_dims(np.array(char_com), 0)
        char_struct = np.expand_dims(np.array(char_struct), 0)

        char_img = np.expand_dims(np.array(char_img), 0) # [N, C, H, W], C=1
        char_img = np.expand_dims(np.array(char_img), 0) # [N, C, H, W], C=1

        character_id = self.char_dict.find(tag_char)

        return {'character_id': torch.Tensor([character_id]),
                'writer_id': self.writer_id,
                'img_list': torch.Tensor(self.img_list),
                'char_img': torch.Tensor(char_img),
                'char_com': torch.Tensor(char_com).to(torch.int),
                'char_struct': torch.Tensor(char_struct).to(torch.int)}


from models.model_inference import SDT_Generator, SDT_Generator_AugThick
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from utils.util import coords_render, dxdynp_to_list
import cv2
from PIL import Image
import torchvision.transforms as T

transform = T.ToPILImage()

class My_model:
    def __init__(self, pretrained_model='./Saved/best.pth',
                        save_dir='./results/tra',
                        cfg_file='configs/CONFIG.yml',
                        store_type='img'):

        cfg_from_file(cfg_file)
        assert_and_infer_cfg()

        self.store_type = store_type
        self.save_dir = save_dir
        if self.store_type == 'online' or self.store_type == 'both':
            os.makedirs(os.path.join(self.save_dir, 'online_test'), exist_ok=True)
        if self.store_type == 'img' or self.store_type == 'both':
            os.makedirs(os.path.join(self.save_dir, 'test'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'test_all'), exist_ok=True)
            # os.makedirs(os.path.join(self.save_dir, 'content'), exist_ok=True)

        """build model architecture"""
        if 'best_thick_aug' in pretrained_model:
            self.model = SDT_Generator_AugThick(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
                    num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
                    wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
                    gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
        else:
            self.model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
                    num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
                    wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
                    gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print('load pretrained model from {}'.format(pretrained_model))

        self.model.eval()

    def __call__(self, my_data, text_content='罢班橙'):
        image_list = []

        for tag_char in text_content:
            with torch.no_grad():
                # prepare input
                data = my_data(tag_char)
                character_id, writer_id, img_list, char_com, char_struct, char_img = data['character_id'].long().cuda(), \
                                                            data['writer_id'], \
                                                            data['img_list'].cuda(), \
                                                            data['char_com'].cuda(), \
                                                            data['char_struct'].cuda(), \
                                                            data['char_img'].cuda()

                preds = self.model.inference(img_list, char_img, char_com, char_struct, 300)

                bs = character_id.shape[0]
                SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
                preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
                preds = preds.detach().cpu().numpy()

                if self.store_type == 'online' or self.store_type == 'both':
                    pred, _ = dxdynp_to_list(preds[0])
                    os.makedirs(os.path.join(self.save_dir, 'online_test', str(writer_id)), exist_ok=True)
                    save_path = os.path.join(self.save_dir, 'online_test',
                                str(writer_id), tag_char + '.txt')
                    with open(save_path, 'w') as f:
                        f.write(str(pred))

                if self.store_type == 'img' or self.store_type == 'both':
                    """intends to blur the boundaries of each sample to fit the actual using situations,
                        as suggested in 'Deep imitator: Handwriting calligraphy imitation via deep attention networks'"""
                    sk_pil = coords_render(preds[0], split=True, width=48, height=48, thickness=1, board=0)
                    sk_pil = sk_pil.resize((64, 64)) #(113, 113)

                    image_list.append(sk_pil)

        ###/
        if self.store_type == 'img' or self.store_type == 'both':
            if len(image_list) > 0:
                # Get the size of the images
                widths, heights = zip(*(i.size for i in image_list))

                # Create a new image with the combined width and height
                new_width = sum(widths)
                max_height = max(heights)
                new_image = Image.new('RGB', (new_width, max_height))

                os.makedirs(os.path.join(self.save_dir, 'test', str(writer_id)), exist_ok=True)
                # Paste the images side by side
                x_offset = 0
                for i, image in enumerate(image_list):
                    new_image.paste(image, (x_offset, 0))
                    x_offset += image.size[0]
                    save_path = os.path.join(self.save_dir, 'test',
                                str(writer_id), text_content[i] + '.jpg')
                    image.save(save_path)

                # Save the concatenated image
                save_path = os.path.join(self.save_dir, 'test_all',
                                str(writer_id) + '_' + text_content + '.jpg')

                # try:
                new_image.save(save_path)


import argparse

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CONFIG.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--model', default='./Saved/best.pth',
                        dest='model_path', required=False, help='trained model path')
    parser.add_argument('--save_dir', default='./results/inference_samples',
                        dest='save_dir', required=False, help='path to save generated image')
    parser.add_argument('--style', default='./data/inference_style_samples',
                        dest='style_list', required=False, help='dir of style images')
    parser.add_argument('--store_type', required=False, default='img',
                        dest='store_type', help='online, img, or both')
    parser.add_argument('-c', '--characters', required=False, default='',
                        dest='characters', help='characters you want to generate')
    opt = parser.parse_args()

    my_model = My_model(
        pretrained_model=opt.model_path,
        save_dir=opt.save_dir,
        cfg_file=opt.cfg_file,
        store_type=opt.store_type)

    for style_dir in os.listdir(opt.style_list):
        style_path = os.path.join(opt.style_list, style_dir)
        my_data = DataProcessing(style_path=style_path, save_path=opt.save_dir)
        my_model(my_data=my_data, text_content=opt.characters)
        
