import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from models.loss import SupConLoss, get_pen_loss
from models.model_pre import SDT_Generator
from utils.logger import set_log
from data_loader.loader import ScriptDataset
from trainer.trainer_pre import Trainer
import torch

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED, is_train=True)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)
    """ set dataset"""
    train_dataset = ScriptDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TRAIN.ISTRAIN, cfg.MODEL.NUM_IMGS, split=True)
    print('number of training images: ', len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=256,
                                               shuffle=True,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS)
    
    char_dict = train_dataset.char_dict2
    """ build model, criterion and optimizer"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    ### load checkpoint
    if len(opt.pretrained_model) > 0:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        pass
    if len(opt.content_pretrained_c) > 0:
        model_dict = load_specific_dict(model.content_encoder_c, opt.content_pretrained_c, "feature_ext")
        model.content_encoder_c.load_state_dict(model_dict)
        print('load content pretrained model from {}'.format(opt.content_pretrained_c))
        print(len(model_dict))

    if len(opt.content_pretrained) > 0:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.content_pretrained)
        pretrained_dict = {k[16:]: v for k, v in pretrained_dict.items() if k.startswith('content_encoder.cls_head')}
        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('load content pretrained model from {}'.format(opt.content_pretrained))
    else:
        pass
    if len(opt.style_pretrained) > 0:
        model_dict = torch.load(opt.style_pretrained)
        # modules = ['Feat_Encoder', 'base_encoder', 'writer_head', 'glyph_head']
        count = len('Feat_Encoder') + 1
        pretrained_dict = {k[count:]: v for k, v in model_dict.items() if k.startswith('Feat_Encoder')}
        model.Feat_Encoder.load_state_dict(pretrained_dict)
        count = len('base_encoder') + 1
        pretrained_dict = {k[count:]: v for k, v in model_dict.items() if k.startswith('base_encoder')}
        model.base_encoder.load_state_dict(pretrained_dict)
        count = len('writer_head') + 1
        pretrained_dict = {k[count:]: v for k, v in model_dict.items() if k.startswith('writer_head')}
        model.writer_head.load_state_dict(pretrained_dict)
        count = len('glyph_head') + 1
        pretrained_dict = {k[count:]: v for k, v in model_dict.items() if k.startswith('glyph_head')}
        model.glyph_head.load_state_dict(pretrained_dict)
        print('load style pretrained model from {}'.format(opt.style_pretrained))
        
    criterion = dict(NCE=SupConLoss(contrast_mode='all'), PEN=get_pen_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    """start training iterations"""

    trainer = Trainer(model, criterion, optimizer, train_loader, logs, char_dict, None)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='./Saved/paper/checkpoint-iter199999.pth',
                        dest='pretrained_model', required=False, help='continue to train model')
    parser.add_argument('--style_pretrained', default='',
                        dest='style_pretrained', required=False, help='continue to train style encoder')
    parser.add_argument('--content_pretrained_c', default='',
                        dest='content_pretrained_c', required=False, help='continue to train content encoder')
    parser.add_argument('--content_pretrained', default='',
                        dest='content_pretrained', required=False, help='continue to train content encoder')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CONFIG.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log', default='pretrained',
                        dest='log_name', required=False, help='the filename of log')
    opt = parser.parse_args()
    main(opt)