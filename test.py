import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import ScriptDataset
import pickle
from models.model import SDT_Generator
import tqdm
from utils.util import writeCache, dxdynp_to_list, coords_render
import lmdb
import csv

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """setup data_loader instances"""
    test_dataset = ScriptDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS, test_dataset=opt.dataset, split=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    print('number of testing images: ', len(test_dataset))
    # char_dict = test_dataset.char_dict2
    # writer_dict = test_dataset.writer_dict
    char_dict = pickle.load(open(f'./data/CHINESE/{opt.char_dict}.pkl', 'rb'))
    all_writer_dict = pickle.load(open('./data/CHINESE/writer_dict_t_only_all_split.pkl', 'rb'))
    writer_dict = all_writer_dict['test_writer']

    ###/
    if opt.store_type == 'online' or opt.store_type == 'both':

        os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
        test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    ###/
    if opt.store_type == 'img' or opt.store_type == 'both':
        os.makedirs(os.path.join(opt.save_dir, 'test2'), exist_ok=True)
        os.makedirs(os.path.join(opt.save_dir, 'test2_gt'), exist_ok=True)
    char_dict2 = test_dataset.char_dict2
    writer_dict2 = test_dataset.writer_dict

    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

    """build model architecture"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS, stage='2').to('cuda')

    if len(opt.pretrained_model) > 0:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        raise IOError('input the correct checkpoint path')
    model.eval()
 
    """calculate the total batches of generated samples"""
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        ###/
        batch_samples = int(opt.sample_size)*10/cfg.TRAIN.IMS_PER_BATCH

    batch_num, num_count= 0, 0
    
    ###/
    if opt.store_type == 'img' or opt.store_type == 'both':
        output_csv = []
        cnt = 0
        tmp_path = './test2/'

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            batch_num += 1
            if batch_num > batch_samples:
                break
            else:
                # prepare input
                coords, coords_len, character_id, writer_id, img_list, char_com, char_struct, char_img = data['coords'].cuda(), \
                    data['coords_len'].cuda(), \
                    data['character_id'].long().cuda(), \
                    data['writer_id'].long().cuda(), \
                    data['img_list'].cuda(), \
                    data['char_com'].cuda(), \
                    data['char_struct'].cuda(), \
                    data['char_img'].cuda()
                character = [char_dict2[c.item()] for c in character_id]

                preds = model.inference(img_list, char_img, char_com, char_struct, 300)
                bs = character_id.shape[0]
                SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
                preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
                preds = preds.detach().cpu().numpy()

                test_cache = {}
                coords = coords.detach().cpu().numpy()
                if opt.store_type == 'online' or opt.store_type == 'both':
                    for i, pred in enumerate(preds):
                        pred, _ = dxdynp_to_list(preds[i])
                        coord, _ = dxdynp_to_list(coords[i])
                        new_character_id = char_dict.find(char_dict2[character_id[i].item()])
                        new_writer_id = writer_dict[list(writer_dict2.keys())[list(writer_dict2.values()).index(writer_id[i].item())]]
                        data = {'coordinates': pred, 'writer_id': new_writer_id,
                                'character_id': new_character_id, 'coords_gt':coord}
                        data_byte = pickle.dumps(data)
                        data_id = str(num_count).encode('utf-8')
                        test_cache[data_id] = data_byte
                        num_count += 1
                    test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
                    writeCache(test_env, test_cache)
                if opt.store_type == 'img' or opt.store_type == 'both':
                    for i, pred in enumerate(preds):
                        """intends to blur the boundaries of each sample to fit the actual using situations,
                            as suggested in 'Deep imitator: Handwriting calligraphy imitation via deep attention networks'"""
                        sk_pil = coords_render(preds[i], split=True, width=48, height=48, thickness=1, board=0)
                        sk_pil = sk_pil.resize((113, 113))

                        new_writer_id = writer_dict[list(writer_dict2.keys())[list(writer_dict2.values()).index(writer_id[i].item())]]
                        character = char_dict2[character_id[i].item()]
                        save_path = os.path.join(opt.save_dir, 'test2',
                                        str(new_writer_id) + '_' + character+'.jpg')
                                        
                        sk_pil_gt = coords_render(coords[i], split=True, width=48, height=48, thickness=1, board=0) 
                        sk_pil_gt = sk_pil_gt.resize((113, 113))
                        save_path_gt = os.path.join(opt.save_dir, 'test2_gt',
                                        str(new_writer_id) + '_' + character+'.jpg')
                        try:
                            sk_pil.save(save_path)
                            sk_pil_gt.save(save_path_gt)
                        except:
                            print('error. %s, %s, %s' % (save_path, str(writer_id[i].item()), character))

                        ###/
                        output_csv.append({
                            '': cnt,
                            'image': tmp_path + str(writer_id[i].item()) + '_' + character+'.jpg',
                            'label': character
                        })
                        cnt += 1

                else:
                    raise NotImplementedError('only support online or img format')

    ###/
    if opt.store_type == 'img' or opt.store_type == 'both':
        with open(os.path.join(opt.save_dir, 'test2.csv'), 'w', newline='') as csvfile:
            fieldnames = ['', 'image', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for d in output_csv:
                writer.writerow(d)

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CONFIG.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--save_dir', dest='save_dir', default='Generated/result', help='target dir for storing the generated characters')
    parser.add_argument('--model', dest='pretrained_model', default='', required=False, help='continue train model')
    parser.add_argument('--store_type', dest='store_type', required=False, default='online', help='online, img, or both')
    parser.add_argument('--sample_size', dest='sample_size', default='200', required=False, help='randomly generate a certain number of characters for each writer')
    parser.add_argument('--dataset', dest='dataset', type=int, default=0, help='0: seen tra, 1: unseen tra')
    parser.add_argument('-c', '--char_dict', dest='char_dict', default='character_dict_t_only_all', required=False, help='character order')
    opt = parser.parse_args()
    main(opt)
