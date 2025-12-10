import torch
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from models.encoder import Content_TR, Content_TR_C
from einops import rearrange, repeat
from models.gmm import get_seq_from_gmm

'''
the overall architecture of our style-disentangled Transformer (SDT).
the input of our SDT is the gray image with 1 channel.
'''
class SDT_Generator(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_head_layers= 1,
                 wri_dec_layers=2, gly_dec_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, return_intermediate_dec=True):
        super(SDT_Generator, self).__init__()
        ### content ecoder
        self.content_encoder = Content_TR(d_model, num_encoder_layers)
        ###/ 2recogn
        self.cls_head = nn.Linear(d_model, 4808)
        self.seq_head = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.seq_head_fc = nn.Linear(d_model, 9825)

        ###/ mix
        self.content_encoder_c = Content_TR_C(d_model, num_encoder_layers)
        mix_layer = CrossModalityLayer(d_model, nhead, dim_feedforward, dropout)
        self.mixer = CrossModalityModule(mix_layer, num_head_layers)

        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)        
        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def random_double_sampling(self, x, ratio=0.25):
        """
        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        anchor_tokens, pos_tokens = int(L*ratio), int(L*2*ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos

    # the shape of style_imgs is [B, 2*N, C, H, W] during training
    ###/ mix
    def forward(self, style_imgs, seq, char_img, char_com, char_struct):
        N = char_img.shape[0]

        ###/ mix
        char_emb_c = self.content_encoder_c(char_com, char_struct) # [4, N, 512]
        char_emb_c = self.add_position(char_emb_c)

        char_emb = self.content_encoder(char_img) # [4, N, 512]
        char_emb = self.add_position(char_emb)
        char_emb += 1 # modality
        ###/ mix
        char_emb, char_emb_c = self.mixer(char_emb, char_emb_c)
        char_emb_c = torch.mean(char_emb_c, 0) #[N, 512]

        char_emb = torch.mean(char_emb, 0) #[N, 512]

        char_cls = self.cls_head(char_emb)
        # Initialize decoder hidden state and cell state
        decoder_hidden = torch.zeros(2, N, 512).to(char_emb_c)
        decoder_cell = torch.zeros(2, N, 512).to(char_emb_c)
        # Generate output sequence
        seq_cls = []
        for t in range(31):
            decoder_output, (decoder_hidden, decoder_cell) = self.seq_head(char_emb_c.unsqueeze(1), (decoder_hidden, decoder_cell))
            output = self.seq_head_fc(decoder_output.squeeze(1))
            seq_cls.append(output)
            decoder_input = output
        seq_cls = torch.stack(seq_cls, dim=1)

        return char_cls, seq_cls

    # style_imgs: [B, N, C, H, W]
    def inference(self, style_imgs, char_img, max_len):
        batch_size, num_imgs, in_planes, h, w = style_imgs.shape
        # [B, N, C, H, W] -> [B*N, C, H, W]
        style_imgs = style_imgs.view(-1, in_planes, h, w)
        # [B*N, 1, 64, 64] -> [B*N, 512, 2, 2]
        style_embe = self.Feat_Encoder(style_imgs)
        FEAT_ST = style_embe.reshape(batch_size*num_imgs, 512, -1).permute(2, 0, 1)  # [4, B*N, C]
        FEAT_ST_ENC = self.add_position(FEAT_ST)  # [4, B*N, C:512]
        memory = self.base_encoder(FEAT_ST_ENC)  # [5, B*N, C]
        memory_writer = self.writer_head(memory)  # [4, B*N, C]
        memory_glyph = self.glyph_head(memory)  # [4, B*N, C]
        memory_writer = rearrange(
            memory_writer, 't (b n) c ->(t n) b c', b=batch_size)  # [4*N, B, C]
        memory_glyph = rearrange(
            memory_glyph, 't (b n) c -> (t n) b c', b=batch_size)  # [4*N, B, C]

        char_emb = self.content_encoder(char_img)
        char_emb = torch.mean(char_emb, 0) #[N, 256]
        src_tensor = torch.zeros(max_len + 1, batch_size, 512).to(char_emb)
        pred_sequence = torch.zeros(max_len, batch_size, 5).to(char_emb)
        src_tensor[0] = char_emb
        tgt_mask = generate_square_subsequent_mask(sz=max_len + 1).to(char_emb)
        for i in range(max_len):
            src_tensor[i] = self.add_position(src_tensor[i], step=i)

            wri_hs = self.wri_decoder(
                src_tensor, memory_writer, tgt_mask=tgt_mask)
            hs = self.gly_decoder(wri_hs[-1], memory_glyph, tgt_mask=tgt_mask)

            output_hid = hs[-1][i]
            gmm_pred = self.EmbtoSeq(output_hid)
            pred_sequence[i] = get_seq_from_gmm(gmm_pred)
            pen_state = pred_sequence[i, :, 2:]
            seq_emb = self.SeqtoEmb(pred_sequence[i])
            src_tensor[i + 1] = seq_emb
            if sum(pen_state[:, -1]) == batch_size:
                break
            else:
                pass
        return pred_sequence.transpose(0, 1)  # N, T, C        

'''
project the handwriting sequences to the transformer hidden space
'''
class SeqtoEmb(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(5, 256)
        self.fc_2 = nn.Linear(256, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x

'''
project the transformer hidden space to handwriting sequences
'''
class EmbtoSeq(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 123)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


''' 
generate the attention mask, i.e. [[0, inf, inf],
                                   [0, 0, inf],
                                   [0, 0, 0]].
The masked positions are filled with float('-inf').
Unmasked positions are filled with float(0.0).                                     
'''
def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask