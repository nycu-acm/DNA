import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
from models.gmm import get_mixture_coef, get_seq_from_gmm
import os
import datetime
import sys
from utils.util import coords_render, dxdynp_to_list, corrds2xys
from PIL import Image
###/
from fastdtw import fastdtw
import pickle
import numpy as np

from torch.nn import CrossEntropyLoss


class Trainer:
    def __init__(self, model, criterion, optimizer, data_loader, 
                logs, char_dict, valid_data_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.char_dict = char_dict
        self.valid_data_loader = valid_data_loader
        self.nce_criterion = criterion['NCE']
        self.pen_criterion = criterion['PEN']
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.cls_criterion = CrossEntropyLoss(ignore_index=-1)
      
    def _train_iter(self, data, step):
        self.model.train()
        prev_time = time.time()
        # prepare input
        coords, coords_len, character_id, character_id1, writer_id, img_list, char_com, char_struct, char_img, char_decom_idx = data['coords'].cuda(), \
            data['coords_len'].cuda(), \
            data['character_id'].long().cuda(), \
            data['character_id1'].long().cuda(), \
            data['writer_id'].long().cuda(), \
            data['img_list'].cuda(), \
            data['char_com'].cuda(), \
            data['char_struct'].cuda(), \
            data['char_img'].cuda(), \
            data['char_decom_idx'].long().cuda()
        
        # forward
        input_seq = coords[:, 1:-1]
        pred_char, pred_seq = self.model(img_list, input_seq, char_img, char_com, char_struct)
        
        loss_c = self.cls_criterion(pred_char, character_id1)
        pred_seq = pred_seq.view(-1, pred_seq.shape[-1])
        char_decom_idx = char_decom_idx.view(-1)
        loss_s = self.cls_criterion(pred_seq, char_decom_idx)

        loss = loss_c + loss_s

        # backward and update trainable parameters
        self.model.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        # log file
        # loss_dict = {"loss": loss.item()}
        loss_dict = {"loss_con":loss_c.item(), "loss_con_seq":loss_s.item()}
        self.tb_summary.add_scalars("loss", loss_dict, step)
        iter_left = cfg.SOLVER.MAX_ITER - step
        time_left = datetime.timedelta(
                    seconds=iter_left * (time.time() - prev_time))
        self._progress(step, [loss.item(), loss_c.item(), loss_s.item()], time_left)

        ###/
        del data
        
        torch.cuda.empty_cache()

        ###/
        return loss.item()

    def _valid_iter(self, step):
        self.model.eval()
        print('loading test dataset, the number is', len(self.valid_data_loader))
        try:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        # prepare input
        coords, coords_len, character_id, writer_id, img_list, char_img, char_decom_idx = test_data['coords'].cuda(), \
            test_data['coords_len'].cuda(), \
            test_data['character_id'].long().cuda(), \
            test_data['writer_id'].long().cuda(), \
            test_data['img_list'].cuda(), \
            test_data['char_img'].cuda(), \
            test_data['char_decom_idx'].long().cuda()
         # forward
        with torch.no_grad():
            preds = self.model.inference(img_list, char_img, 120)
            bs = character_id.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the first token
            preds = preds.cpu().numpy()
            gt_coords = coords.cpu().numpy()  # [N, T, C]
            self._vis_genarate_samples(gt_coords, preds, character_id, step)

            ###/
            euclidean = lambda x, y: np.sqrt(sum((x - y) ** 2))
            fast_norm_dtw_len, total_num = 0, 0
            for i, pred in enumerate(preds):
                pred, _ = dxdynp_to_list(preds[i])
                coord, _ = dxdynp_to_list(gt_coords[i])
                # evaluate
                pred, coord = corrds2xys(pred), corrds2xys(coord)
                pred_len = pred.shape[0]
                gt_len = coord.shape[0]
                pred_valid = torch.from_numpy(pred[:pred_len])
                gt_valid = torch.from_numpy(coord[:gt_len])
                # Convert relative coordinates into absolute coordinates
                seq_1 = torch.cumsum(gt_valid[:, :2], dim=0)
                if pred_valid.shape[0] > 0:
                    seq_2 = torch.cumsum(pred_valid[:, :2], dim=0)
                else:
                    seq_2 = torch.tensor([[0, 0], [0, 0]]).to(pred_valid)
                # DTW between paired real and fake online characters
                fast_d, _ = fastdtw(seq_1, seq_2, dist= euclidean)
                fast_norm_dtw_len += (fast_d/gt_len)
            total_num += len(preds)
            avg_fast_norm_dtw_len = fast_norm_dtw_len/total_num
            print(f"the avg fast_norm_len_dtw is {avg_fast_norm_dtw_len}")
        # return avg_fast_norm_dtw_len

    def train(self):
        """start training iterations"""    
        best_loss = 100
        # best_val_loss = 100

        train_loader_iter = iter(self.data_loader)
        for step in range(cfg.SOLVER.MAX_ITER):
            try:
                data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.data_loader)
                data = next(train_loader_iter)

            loss = self._train_iter(data, step)

            if self.valid_data_loader is not None:
                if (step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(step)
            else:
                pass
            if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self._save_checkpoint(step, loss, None, best=loss<best_loss)#, best_val=dtw_loss<best_val_loss)
                if loss < best_loss:
                    best_loss = loss

            else:
                pass

    
    def _progress(self, step, loss, time_left):
        terminal_log = 'iter:%d ' % step
        terminal_log += '%s:[%.3f, %.3f, %.3f] ' % ('loss', loss[0], loss[1], loss[2])
        terminal_log += 'ETA:%s\r\n' % str(time_left)
        sys.stdout.write(terminal_log)

    def _save_checkpoint(self, step, loss, val_loss, best=False):#), best_val=False):
        if best:
            model_path = '{}/best.pth'.format(self.save_model_dir)
            torch.save(self.model.state_dict(), model_path)
            print('save model to {}'.format(model_path))
            model_path = '{}/best_iter.txt'.format(self.save_model_dir)
            with open(model_path, 'w') as f:
                f.write(f'{step}\t{loss}')
        model_path = '{}/last.pth'.format(self.save_model_dir)
        torch.save(self.model.state_dict(), model_path)
        print('save model to {}'.format(model_path))
        model_path = '{}/last_iter.txt'.format(self.save_model_dir)
        with open(model_path, 'w') as f:
            f.write(f'{step}\t{loss}')

    def _vis_genarate_samples(self, gt_coords, preds, character_id, step):
        for i, _ in enumerate(gt_coords):
            gt_img = coords_render(gt_coords[i], split=True, width=64, height=64, thickness=1)
            pred_img = coords_render(preds[i], split=True, width=64, height=64, thickness=1)
            example_img = Image.new("RGB", (cfg.TEST.IMG_W * 2, cfg.TEST.IMG_H),
                                    (255, 255, 255))
            example_img.paste(pred_img, (0, 0)) # gererated character
            example_img.paste(gt_img, (cfg.TEST.IMG_W, 0)) # gt character
            character = self.char_dict[character_id[i].item()]
            save_path = os.path.join(self.save_sample_dir, 'ite.' + str(step//100000)
                 + '-'+ str(step//100000 + 100000), character + '_' + str(step) + '_.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                example_img.save(save_path)
            except:
                print('error. %s, %s' % (save_path, character))