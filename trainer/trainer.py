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
###/ stroke
from torchvision.ops import generalized_box_iou_loss
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss


###/ stroke
def get_boxes(coords):
    coordinates = coords.clone()
    coordinates[:, 0] = torch.cumsum(coordinates[:, 0], dim=0)
    coordinates[:, 1] = torch.cumsum(coordinates[:, 1], dim=0)

    ids = torch.where(coordinates[:, -1] == 1)[0]
    if len(ids) < 1:  ### if not exist [0, 0, 1]
        ids = torch.where(coordinates[:, 3] == 1)[0] + 1
        if len(ids) < 1: ### if not exist [0, 1, 0]
            ids = [len(coordinates)]
            xys_split = torch.split(coordinates, ids, dim=0)[:-1] # remove the blank list
        else:
            last = [coordinates.shape[0]-ids[-1]]
            ids = [ids[0]] + [ids[i+1]-ids[i] for i in range(ids.shape[0]-1)] + last
            xys_split = torch.split(coordinates, ids, dim=0)[:-1]
    else:  ### if exist [0, 0, 1]
        last = [coordinates.shape[0]-ids[-1]]
        ids = [ids[0]] + [ids[i+1]-ids[i] for i in range(ids.shape[0]-1)] + last
        remove_end = torch.split(coordinates, ids, dim=0)[0]
        ids = torch.where(remove_end[:, 3] == 1)[0] + 1 ### break in [0, 1, 0]
        if len(ids) > 0:
            last = [remove_end.shape[0]-ids[-1]]
            ids = [ids[0]] + [ids[i+1]-ids[i] for i in range(ids.shape[0]-1)] + last
            xys_split = torch.split(remove_end, ids, dim=0)[:-1]
        else:
            ids = [len(remove_end)]
            xys_split = torch.split(remove_end, ids, dim=0)[:-1]

    boxes = torch.zeros((len(xys_split), 4)).to(coordinates)
    min_x = torch.Tensor([float("Inf")]).to(coordinates)
    min_y = torch.Tensor([float("Inf")]).to(coordinates)
    max_x = torch.Tensor([-float("Inf")]).to(coordinates)
    max_y = torch.Tensor([-float("Inf")]).to(coordinates)
    for i in range(len(xys_split)):
        xs, ys = xys_split[i][:, 0], xys_split[i][:, 1]
        x1, x2 = torch.min(xs), torch.max(xs)
        y1, y2 = torch.min(ys), torch.max(ys)
        min_x = torch.min(x1, min_x)
        max_x = torch.max(x2, max_x)
        min_y = torch.min(y1, min_y)
        max_y = torch.max(y2, max_y)
        boxes[i] = torch.stack([x1, y1, x2, y2])
    original_size = torch.max(max_x - min_x, max_y - min_y)
    boxes[:, ::2] = (boxes[:, ::2] - min_x) / original_size * 54 + 54
    boxes[:, 1::2] = (boxes[:, ::2] - min_y) / original_size * 54 + 54
    boxes = torch.round(boxes)

    return boxes, len(xys_split) # [L, 4]

class Trainer:
    def __init__(self, model, criterion, optimizer, data_loader, 
                logs, char_dict, valid_data_loader=None, stage='1'):
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
        self.stage = stage
        self.cls_criterion = CrossEntropyLoss(ignore_index=-1)
      
    def _train_iter_stage1(self, data, step):
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
        preds, nce_emb, nce_emb_patch, pred_char, pred_seq = self.model(img_list, input_seq, char_img, char_com, char_struct)
        
        preds2 = torch.zeros(preds.shape[0], preds.shape[1], 5).to(preds)
        for i in range(preds.shape[1]):
            preds2[:, i] = get_seq_from_gmm(preds[:, i])

        # calculate loss
        gt_coords = coords[:, 1:, :]
        nce_loss_writer = self.nce_criterion(nce_emb, labels=writer_id)
        nce_loss_glyph = self.nce_criterion(nce_emb_patch)
        preds = preds.view(-1, 123)
        gt_coords = gt_coords.reshape(-1, 5)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = get_mixture_coef(preds)
        moving_loss_all, state_loss = self.pen_criterion(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, \
                                      o_corr, o_pen_logits, gt_coords[:,0].unsqueeze(-1), gt_coords[:,1].unsqueeze(-1), gt_coords[:,2:])
        moving_loss = torch.sum(moving_loss_all) / torch.sum(coords_len)
        pen_loss = moving_loss + 2*state_loss

        loss_c = self.cls_criterion(pred_char, character_id1)
        pred_seq = pred_seq.view(-1, pred_seq.shape[-1])
        char_decom_idx = char_decom_idx.view(-1)
        loss_s = self.cls_criterion(pred_seq, char_decom_idx)

        loss = pen_loss + nce_loss_writer + nce_loss_glyph + 0.5*loss_c + 0.5*loss_s

        # backward and update trainable parameters
        self.model.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        # log file
        # loss_dict = {"loss": loss.item()}
        loss_dict = {"pen_loss": pen_loss.item(), "moving_loss": moving_loss.item(),
                    "state_loss": state_loss.item(), "nce_loss_writer":nce_loss_writer.item(), 
                    "nce_loss_glyph":nce_loss_glyph.item(), "loss_con":loss_c.item(), "loss_con_seq":loss_s.item()}
        self.tb_summary.add_scalars("loss", loss_dict, step)
        iter_left = cfg.SOLVER.MAX_ITER - step
        time_left = datetime.timedelta(
                    seconds=iter_left * (time.time() - prev_time))
        self._progress(step, loss.item(), time_left)

        ###/
        del data, preds, preds2
        
        torch.cuda.empty_cache()

        return loss.item()

    def _train_iter_stage2(self, data, step):
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
        preds, nce_emb, nce_emb_patch = self.model(img_list, input_seq, char_img, char_com, char_struct)
        
        preds2 = torch.zeros(preds.shape[0], preds.shape[1], 5).to(preds)
        for i in range(preds.shape[1]):
            preds2[:, i] = get_seq_from_gmm(preds[:, i])

        # calculate loss
        gt_coords = coords[:, 1:, :]
        nce_loss_writer = self.nce_criterion(nce_emb, labels=writer_id)
        nce_loss_glyph = self.nce_criterion(nce_emb_patch)
        preds = preds.view(-1, 123)
        gt_coords = gt_coords.reshape(-1, 5)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = get_mixture_coef(preds)
        moving_loss_all, state_loss = self.pen_criterion(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, \
                                      o_corr, o_pen_logits, gt_coords[:,0].unsqueeze(-1), gt_coords[:,1].unsqueeze(-1), gt_coords[:,2:])
        moving_loss = torch.sum(moving_loss_all) / torch.sum(coords_len)
        pen_loss = moving_loss + 2*state_loss

        bs = character_id.shape[0]
        SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds2)
        preds2 = torch.cat((SOS, preds2), 1)  # add the first token
        for i in range(preds2.shape[0]):
            preds_boxes, preds_len = get_boxes(preds2[i])
            gt_boxes, gt_len = get_boxes(coords[i])
            boxes = pad_sequence([preds_boxes, gt_boxes], batch_first=True) # [2, L, 4]
            if i == 0:
                all_boxes = boxes
            else:
                all_boxes = torch.cat((all_boxes, boxes), dim=1)
        loss_b = generalized_box_iou_loss(all_boxes[0], all_boxes[1], reduction='mean')

        loss = pen_loss + nce_loss_writer + nce_loss_glyph + loss_b

        # backward and update trainable parameters
        self.model.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        # log file
        # loss_dict = {"loss": loss.item()}
        loss_dict = {"pen_loss": pen_loss.item(), "moving_loss": moving_loss.item(),
                    "state_loss": state_loss.item(), "nce_loss_writer":nce_loss_writer.item(), 
                    "nce_loss_glyph":nce_loss_glyph.item(), "loss_box":loss_b.item()}#, "loss_con_one":loss_c.item(), "loss_con_seq":loss_s.item()}
        self.tb_summary.add_scalars("loss", loss_dict, step)
        iter_left = cfg.SOLVER.MAX_ITER - step
        time_left = datetime.timedelta(
                    seconds=iter_left * (time.time() - prev_time))
        self._progress(step, loss.item(), time_left)

        ###/
        del data, preds
        
        torch.cuda.empty_cache()

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
        coords, coords_len, character_id, writer_id, img_list, char_com, char_struct, char_img, char_decom_idx = test_data['coords'].cuda(), \
            test_data['coords_len'].cuda(), \
            test_data['character_id'].long().cuda(), \
            test_data['writer_id'].long().cuda(), \
            test_data['img_list'].cuda(), \
            test_data['char_com'].cuda(), \
            test_data['char_struct'].cuda(), \
            test_data['char_img'].cuda(), \
            test_data['char_decom_idx'].long().cuda()
         # forward
        with torch.no_grad():
            preds = self.model.inference(img_list, char_img, char_com, char_struct, 120)
            bs = character_id.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the first token
            preds = preds.cpu().numpy()
            gt_coords = coords.cpu().numpy()  # [N, T, C]
            self._vis_genarate_samples(gt_coords, preds, character_id, step)


    def train(self):
        """start training iterations"""    
        ###/
        best_loss = 100
        # best_val_loss = 100

        train_loader_iter = iter(self.data_loader)
        for step in range(cfg.SOLVER.MAX_ITER):
            try:
                data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.data_loader)
                data = next(train_loader_iter)
            ###/
            if self.stage == '1':
                loss = self._train_iter_stage1(data, step)
            else:
                loss = self._train_iter_stage2(data, step)

            if self.valid_data_loader is not None:
                if (step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(step)
            else:
                dtw_loss = None
                pass
            if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self._save_checkpoint(step, loss, None, best=loss<best_loss, best_val=False)
                if loss < best_loss:
                    best_loss = loss

            else:
                pass

    def _progress(self, step, loss, time_left):
        terminal_log = 'iter:%d ' % step
        terminal_log += '%s:[%.3f] ' % ('loss', loss)
        terminal_log += 'ETA:%s\r\n' % str(time_left)
        sys.stdout.write(terminal_log)

    def _save_checkpoint(self, step, loss, val_loss, best=False, best_val=False):
        if best:
            model_path = '{}/best.pth'.format(self.save_model_dir)
            torch.save(self.model.state_dict(), model_path)
            print('save model to {}'.format(model_path))
            model_path = '{}/best_iter.txt'.format(self.save_model_dir)
            with open(model_path, 'w') as f:
                f.write(f'{step}\t{loss}')
        if best_val:
            model_path = '{}/val_best.pth'.format(self.save_model_dir)
            torch.save(self.model.state_dict(), model_path)
            print('save model to {}'.format(model_path))
            model_path = '{}/val_best_iter.txt'.format(self.save_model_dir)
            with open(model_path, 'w') as f:
                f.write(f'{step}\t{val_loss}')
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