import torch
import torch.nn as nn
from .. import builder
from .. import loss
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
class MLRL(nn.Module):
    def __init__(self, cfg):
        super(MLRL, self).__init__()
        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)
        if ('/continue/' in cfg.lightning.ckpt) or (('cls' not in cfg.ablation) and ('_cls' not in cfg.lightning.ckpt)) or (('cls' in cfg.ablation) and ('Text' in cfg.ablation) and ('_cls' not in cfg.lightning.ckpt)):
            self.img_decoder = builder.build_img_decoder(cfg)
        self._calc_inner_g_loss = loss.mlrl_loss.inner_loss_g
        self._calc_inner_l_loss = loss.mlrl_loss.inner_loss_l
        self._calc_inner_side_loss = loss.mlrl_loss.inner_loss_side_g
        self.LVTA_loss_l=  loss.lvta_loss.l_loss
        self.LVTA_loss_g = loss.lvta_loss.g_loss
        self.lva_lta_loss_l = loss.lvta_loss.lva_lta_l_loss
        self.rec_loss = loss.mlrl_loss.rec_loss
        self.l_loss = loss.mlrl_loss.l_loss
        self.g_loss = loss.mlrl_loss.g_loss
        self.l_loss_weight = self.cfg.model.mlrl.l_loss_weight
        self.g_loss_weight = self.cfg.model.mlrl.g_loss_weight
        self.mlm = CrossEntropyLoss()
        self.temp1 = self.cfg.model.mlrl.temp1
        self.temp2 = self.cfg.model.mlrl.temp2
        self.temp3 = self.cfg.model.mlrl.temp3
        self.batch_size = self.cfg.train.batch_size
        self.flag = torch.ones(self.batch_size * 4).cuda()
        self.celoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bceloss = nn.BCEWithLogitsLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids, p_caption_ids, p_attention_mask, p_token_type_ids):
        text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2 = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2 = self.text_encoder(
            p_caption_ids, p_attention_mask, p_token_type_ids
        )
        return text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2

    def image_encoder_forward(self, batch):

        f_lcc = self.img_encoder(batch['imgslcc'],batch['p_imgslcc'])  # 【2，c，w, h】
        f_lmlo = self.img_encoder(batch['imgslmlo'],batch['p_imgslmlo'])  # 【2，c，w, h】
        f_rcc = self.img_encoder(batch['imgsrcc'],batch['p_imgsrcc'])  # 【2，c，w, h】
        f_rmlo = self.img_encoder(batch['imgsrmlo'],batch['p_imgsrmlo'])  # 【2，c，w, h】

        dif_size = int (f_lcc.previous_embedding_g.shape[1] )
        t_img_feat_l = []
        for i in range (f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_l[:,dif_size:,:,:], 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_l[:,dif_size:,:,:], 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_l[:,dif_size:,:,:], 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_l[:,dif_size:,:,:], 1, dim=0)[i]
            t_img_feat_l.append(temp_lcc)
            t_img_feat_l.append(temp_lmlo)
            t_img_feat_l.append(temp_rcc)
            t_img_feat_l.append(temp_rmlo)
        t_img_feat_l = torch.stack(t_img_feat_l)
        t_img_feat_l = torch.squeeze(t_img_feat_l,dim=1)

        img_feat_l_x1 = []
        for i in range (f_lcc.x1.shape[0]):
            temp_lcc = torch.split(f_lcc.x1, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.x1, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.x1, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.x1, 1, dim=0)[i]
            img_feat_l_x1.append(temp_lcc)
            img_feat_l_x1.append(temp_lmlo)
            img_feat_l_x1.append(temp_rcc)
            img_feat_l_x1.append(temp_rmlo)
        img_feat_l_x1 = torch.stack(img_feat_l_x1)
        img_feat_l_x1 = torch.squeeze(img_feat_l_x1,dim=1)

        img_feat_l_x2 = []
        for i in range (f_lcc.x2.shape[0]):
            temp_lcc = torch.split(f_lcc.x2, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.x2, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.x2, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.x2, 1, dim=0)[i]
            img_feat_l_x2.append(temp_lcc)
            img_feat_l_x2.append(temp_lmlo)
            img_feat_l_x2.append(temp_rcc)
            img_feat_l_x2.append(temp_rmlo)
        img_feat_l_x2 = torch.stack(img_feat_l_x2)
        img_feat_l_x2 = torch.squeeze(img_feat_l_x2,dim=1)

        img_feat_l_x3 = []
        for i in range (f_lcc.x3.shape[0]):
            temp_lcc = torch.split(f_lcc.x3, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.x3, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.x3, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.x3, 1, dim=0)[i]
            img_feat_l_x3.append(temp_lcc)
            img_feat_l_x3.append(temp_lmlo)
            img_feat_l_x3.append(temp_rcc)
            img_feat_l_x3.append(temp_rmlo)
        img_feat_l_x3 = torch.stack(img_feat_l_x3)
        img_feat_l_x3 = torch.squeeze(img_feat_l_x3,dim=1)

        t_img_feat_g = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_g[:,dif_size:], 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_g[:,dif_size:], 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_g[:,dif_size:], 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_g[:,dif_size:], 1, dim=0)[i]
            t_img_feat_g.append(temp_lcc)
            t_img_feat_g.append(temp_lmlo)
            t_img_feat_g.append(temp_rcc)
            t_img_feat_g.append(temp_rmlo)
        t_img_feat_g = torch.stack(t_img_feat_g)
        t_img_feat_g = torch.squeeze(t_img_feat_g, dim=1)

        c_img_feat_g = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_g[:,0:dif_size], 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_g[:,0:dif_size], 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_g[:,0:dif_size], 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_g[:,0:dif_size], 1, dim=0)[i]
            c_img_feat_g.append(temp_lcc)
            c_img_feat_g.append(temp_lmlo)
            c_img_feat_g.append(temp_rcc)
            c_img_feat_g.append(temp_rmlo)
        c_img_feat_g = torch.stack(c_img_feat_g)
        c_img_feat_g = torch.squeeze(c_img_feat_g, dim=1)

        c_img_feat_l = []
        for i in range(f_lcc.cad_patch_embeddings_l.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_l[:, 0:dif_size,:,:], 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_l[:, 0:dif_size,:,:], 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_l[:, 0:dif_size,:,:], 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_l[:, 0:dif_size,:,:], 1, dim=0)[i]
            c_img_feat_l.append(temp_lcc)
            c_img_feat_l.append(temp_lmlo)
            c_img_feat_l.append(temp_rcc)
            c_img_feat_l.append(temp_rmlo)
        c_img_feat_l = torch.stack(c_img_feat_l)
        c_img_feat_l = torch.squeeze(c_img_feat_l, dim=1)

        p_img_feat_g =[]
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.previous_embedding_g, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.previous_embedding_g, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.previous_embedding_g, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.previous_embedding_g, 1, dim=0)[i]
            p_img_feat_g.append(temp_lcc)
            p_img_feat_g.append(temp_lmlo)
            p_img_feat_g.append(temp_rcc)
            p_img_feat_g.append(temp_rmlo)
        p_img_feat_g = torch.stack(p_img_feat_g)
        p_img_feat_g = torch.squeeze(p_img_feat_g,dim=1)

        p_img_feat_l=[]
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.previous_embedding_l, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.previous_embedding_l, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.previous_embedding_l, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.previous_embedding_l, 1, dim=0)[i]
            p_img_feat_l.append(temp_lcc)
            p_img_feat_l.append(temp_lmlo)
            p_img_feat_l.append(temp_rcc)
            p_img_feat_l.append(temp_rmlo)
        p_img_feat_l = torch.stack(p_img_feat_l)
        p_img_feat_l = torch.squeeze(p_img_feat_l,dim=1)


        cad_img_feat_g=[]
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_g, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_g, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_g, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_g, 1, dim=0)[i]

            cad_img_feat_g.append(temp_lcc)
            cad_img_feat_g.append(temp_lmlo)
            cad_img_feat_g.append(temp_rcc)
            cad_img_feat_g.append(temp_rmlo)
        cad_img_feat_g = torch.stack(cad_img_feat_g)
        cad_img_feat_g = torch.squeeze(cad_img_feat_g,dim=1)

        cad_img_feat_l =[]
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_l, 1, dim=0)[i]
            temp_lmlo = torch.split(f_lmlo.cad_patch_embeddings_l, 1, dim=0)[i]
            temp_rcc = torch.split(f_rcc.cad_patch_embeddings_l, 1, dim=0)[i]
            temp_rmlo = torch.split(f_rmlo.cad_patch_embeddings_l, 1, dim=0)[i]

            cad_img_feat_l.append(temp_lcc)
            cad_img_feat_l.append(temp_lmlo)
            cad_img_feat_l.append(temp_rcc)
            cad_img_feat_l.append(temp_rmlo)
        cad_img_feat_l = torch.stack(cad_img_feat_l)
        cad_img_feat_l = torch.squeeze(cad_img_feat_l, dim=1)

        return t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, img_feat_l_x1, img_feat_l_x2, img_feat_l_x3, f_lcc.logits_cp_t1, f_lcc.logits_d_t1, f_lmlo.logits_cp_t1, f_lmlo.logits_d_t1, f_rcc.logits_cp_t1, f_rcc.logits_d_t1, f_rmlo.logits_cp_t1, f_rmlo.logits_d_t1, f_lcc.logits_cp_t2, f_lcc.logits_d_t2, f_lmlo.logits_cp_t2, f_lmlo.logits_d_t2, f_rcc.logits_cp_t2, f_rcc.logits_d_t2, f_rmlo.logits_cp_t2, f_rmlo.logits_d_t2,     f_lcc.logits_cad_t1, f_lcc.logits_cad_t2, f_lmlo.logits_cad_t1, f_lmlo.logits_cad_t2, f_rcc.logits_cad_t1, f_rcc.logits_cad_t2, f_rmlo.logits_cad_t1, f_rmlo.logits_cad_t2, f_lcc.logits_c_t3,  f_lmlo.logits_c_t3,  f_rcc.logits_c_t3,  f_rmlo.logits_c_t3, f_lcc.logits_diff_t3, f_lmlo.logits_diff_t3, f_rcc.logits_diff_t3, f_rmlo.logits_diff_t3, f_lcc.logits_cad_t3, f_lmlo.logits_cad_t3, f_rcc.logits_cad_t3, f_rmlo.logits_cad_t3
    def image_encoder_forward_forzeroshot(self, curr_img, prev_img):
        # with torch.no_grad():
        f_lcc = self.img_encoder(curr_img, prev_img)  # 【2，c，w, h】

        dif_size = int(f_lcc.previous_embedding_g.shape[1])
        t_img_feat_l = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_l[:, dif_size:, :, :], 1, dim=0)[i]

            t_img_feat_l.append(temp_lcc)

        t_img_feat_l = torch.stack(t_img_feat_l)
        t_img_feat_l = torch.squeeze(t_img_feat_l, dim=1)

        t_img_feat_g = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_g[:, dif_size:], 1, dim=0)[i]

            t_img_feat_g.append(temp_lcc)

        t_img_feat_g = torch.stack(t_img_feat_g)
        t_img_feat_g = torch.squeeze(t_img_feat_g, dim=1)

        c_img_feat_g = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_g[:, 0:dif_size], 1, dim=0)[i]

            c_img_feat_g.append(temp_lcc)

        c_img_feat_g = torch.stack(c_img_feat_g)
        c_img_feat_g = torch.squeeze(c_img_feat_g, dim=1)

        c_img_feat_l = []
        for i in range(f_lcc.cad_patch_embeddings_l.shape[0]):
            temp_lcc = torch.split(f_lcc.cad_patch_embeddings_l[:, 0:dif_size, :, :], 1, dim=0)[i]
            c_img_feat_l.append(temp_lcc)

        c_img_feat_l = torch.stack(c_img_feat_l)
        c_img_feat_l = torch.squeeze(c_img_feat_l, dim=1)

        p_img_feat_g = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.previous_embedding_g, 1, dim=0)[i]
            p_img_feat_g.append(temp_lcc)

        p_img_feat_g = torch.stack(p_img_feat_g)
        p_img_feat_g = torch.squeeze(p_img_feat_g, dim=1)
        # p_img_feat_g = torch.cat((temp_lcc, temp_lmlo, temp_rcc, temp_rmlo),0)
        p_img_feat_l = []
        for i in range(f_lcc.previous_embedding_g.shape[0]):
            temp_lcc = torch.split(f_lcc.previous_embedding_l, 1, dim=0)[i]

            p_img_feat_l.append(temp_lcc)

        p_img_feat_l = torch.stack(p_img_feat_l)
        p_img_feat_l = torch.squeeze(p_img_feat_l, dim=1)

        return c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, t_img_feat_g, t_img_feat_l

    def _calc_l_loss(self, img_emb_l, text_emb_l, sents, label, keyword, flag):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps, classify_loss_l = self.l_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            label,
            keyword,
            flag,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps, classify_loss_l

    def _calc_g_loss(self, img_emb_g, text_emb_g, keyword, flag):
        g_loss0, g_loss1, classify_loss_g = self.g_loss(img_emb_g, text_emb_g, keyword, flag, temp3=self.temp3)
        return g_loss0, g_loss1, classify_loss_g

    def calc_loss(self, split, t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2,  p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, swap_textembedding, p_swap_textembedding,  rec_current, rec_previous, f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1, f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2, f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2, f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1, f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1, f_rmlo_logits_cad_t2, f_lcc_logits_c_t3,  f_lmlo_logits_c_t3,  f_rcc_logits_c_t3,  f_rmlo_logits_c_t3, f_lcc_logits_d_t3, f_lmlo_logits_d_t3, f_rcc_logits_d_t3, f_rmlo_logits_d_t3, f_lcc_logits_cad_t3, f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3, x,  current_epoch):

        # cad,current and dynamic, previous + current:
        if 'cls' not in self.cfg.ablation or 'pre' in self.cfg.ablation:
            cad_l_loss0, cad_l_loss1, attn_maps_cad, coarse_to_fine_loss_l = self._calc_l_loss(cad_img_feat_l, text_emb_l, sents, x['label'], 0, x['flag'])#0-keyword
            cad_g_loss0, cad_g_loss1, coarse_to_fine_loss_g = self._calc_g_loss(cad_img_feat_g, text_emb_g, 0, x['flag'])
            cad_inner_g = self._calc_inner_g_loss(cad_img_feat_g, text_mlp1, text_mlp2, x["imgs"].size(0))
            # cad_inner_l = self._calc_inner_l_loss(cad_img_feat_l, text_mlp1, text_mlp2, x["imgs"].size(0))
            loss_cad_i_r  = cad_l_loss0 + cad_l_loss1 + cad_g_loss0 + cad_g_loss1 + cad_inner_g

            # t_i, current_report_dynamic, previous + current:
            temp_l_loss0, temp_l_loss1, attn_maps_tem, coarse_to_fine_loss_l  = self._calc_l_loss(t_img_feat_l, text_emb_l[:,0:384,:], sents, x['label'], 0, x['flag'])
            temp_g_loss0, temp_g_loss1, coarse_to_fine_loss_g = self._calc_g_loss(t_img_feat_g, text_emb_g[:,0:384], 0, x['flag'])
            temp_inner_g = self._calc_inner_g_loss(t_img_feat_g, text_mlp1, text_mlp2, x["imgs"].size(0))
            # inner_l = self._calc_inner_l_loss(t_img_feat_l, text_mlp1, text_mlp2, x["imgs"].size(0))
            loss_temp_i_r = temp_l_loss0 + temp_l_loss1 + temp_g_loss0 + temp_g_loss1 + temp_inner_g

            # combine(previous_i), previous_report_static
            p_l_loss0, p_l_loss1, attn_maps_pre, coarse_to_fine_loss_l = self._calc_l_loss(p_img_feat_l, p_text_emb_l[:, 384:, :], p_sents, x['label'], 0, x['flag'])
            p_g_loss0, p_g_loss1, coarse_to_fine_loss_g = self._calc_g_loss(p_img_feat_g, p_text_emb_g[:, 384:], 0, x['flag'])
            p_inner_g = self._calc_inner_g_loss(p_img_feat_g, p_text_mlp1, p_text_mlp2, x["imgs"].size(0))
            # p_inner_l = self._calc_inner_l_loss(p_img_feat_l, p_text_mlp1, p_text_mlp2, x["imgs"].size(0))
            loss_p_i_r = p_l_loss0 + p_l_loss1 + p_g_loss0 + p_g_loss1 + p_inner_g
            # combine(current_i), current_report_static
            flag1 = torch.tensor(torch.ones(self.batch_size * 4).cuda(), dtype=torch.int64)
            c_l_loss0, c_l_loss1, attn_maps_curr, coarse_to_fine_loss_l = self._calc_l_loss(c_img_feat_l, text_emb_l[:,384:,:], sents, x['label'], 0, flag1)
            c_g_loss0, c_g_loss1,coarse_to_fine_loss_g = self._calc_g_loss(c_img_feat_g, text_emb_g[:,384:], 0, flag1)
            c_inner_g = self._calc_inner_g_loss(c_img_feat_g, text_mlp1, text_mlp2, x["imgs"].size(0))
            # c_inner_l = self._calc_inner_l_loss(c_img_feat_l, text_mlp1, text_mlp2, x["imgs"].size(0))
            loss_c_i_r = c_l_loss0 + c_l_loss1 + c_g_loss0 + c_g_loss1  + c_inner_g

        # LVTA_loss
        lvta_loss_l, attn_maps_p, attn_maps_c = self.LVTA_loss_l(c_img_feat_l, text_emb_l[:,384:,:], p_img_feat_l, p_text_emb_l[:, 384:, :],[len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents])
        lvta_loss_g, similarities_lvta_c, similarities_lvta_p = self.LVTA_loss_g(c_img_feat_g, p_img_feat_g, text_emb_g[:, 384:], p_text_emb_g[:, 384:])

        lva_lta_l_loss, attn_maps_lva, attn_maps_lta, similarities_lva, similarities_lta = self.lva_lta_loss_l(
            c_img_feat_l.view(c_img_feat_l.shape[0],c_img_feat_l.shape[1], c_img_feat_l.shape[2]*c_img_feat_l.shape[3]),
            p_img_feat_l,
            text_emb_l[:, 384:, :],
            p_text_emb_l[:, 384:, :].view(p_text_emb_l.shape[0], 384, 10, 20),
            [itm for itm in [p_img_feat_l.size(2)*p_img_feat_l.size(3)] for i in range(p_img_feat_l.size(0))],
            [len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents],
        )
        lva_lta_g_loss, similarities_lva, similarities_lta = self.LVTA_loss_g(
            c_img_feat_g,
            text_emb_g[:, 384:],
            p_img_feat_g,
            p_text_emb_g[:, 384:])

        if self.cfg.modality == 'mri':
            if self.cfg.ablation == 'Img_ST_cls_baseline' or self.cfg.ablation == 'Img_ST_cls_proposed':# classification
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                loss = self.celoss(pi, x['label'])
                return loss, pi, x['label'], self.celoss(pi, x['label']), 0, 0, 0, 0, 0, self.celoss(pi, x['label']), 0,0,0
            if self.cfg.ablation == 'Img_MT_cls_baseline':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                prob = self.img_encoder.encoder.fc_Img_MT_b(torch.cat((p_img_feat_g,c_img_feat_g),dim=1))
                loss =  self.celoss(prob, x['label']) + self.celoss(ci, x['label']) +self.celoss(pi, x['label'])
                return loss, prob, x['label'], self.celoss(pi, x['label']), 0, self.celoss(ci,x['label']), 0, 0, 0, self.celoss(prob, x['label']), attn_maps_lva,0,0
            if self.cfg.ablation == 'Img_MT_cls_proposed':  # classification
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                # lva_i = self.img_encoder.encoder.fc_lva(similarities_lva)
                # lva_i = self.img_encoder.encoder.fc_tem_v(t_img_feat_g)
                index_negativeone = torch.nonzero(x['label'] == -1).squeeze()
                similva_diagonal = torch.zeros((similarities_lva.shape[0] - len(index_negativeone), 1)).cuda()
                similva_diagonal_ce = torch.zeros((similarities_lva.shape[0], 1)).cuda()
                bce_label = torch.zeros((similarities_lva.shape[0] - len(index_negativeone), 1)).cuda()
                j = 0
                for i in range(similarities_lva.shape[0]):
                    if i not in index_negativeone:
                        similva_diagonal[j, :] = similarities_lva[i][i]
                        bce_label[j, :] = x['label'][i]
                        j = j + 1
                    similva_diagonal_ce[i, :] = similarities_lva[i][i]
                if len(index_negativeone) == similarities_lva.shape[0]:
                    loss_similva_diagonal = torch.zeros(1).cuda()
                else:
                    labels = torch.ones(bce_label.shape).cuda()
                    loss_similva_diagonal = self.bceloss(similva_diagonal, (labels - bce_label).float())
                similva_diagonal_ = []
                for count in range(16):
                    similva_diagonal_.append(similva_diagonal)
                similva_diagonal_ = torch.stack(similva_diagonal_)
                similva_diagonal_ = similva_diagonal_.view(similva_diagonal.size(0), 16)
                prob = self.img_encoder.encoder.fc_Img_MT_p(torch.cat((p_img_feat_g,c_img_feat_g,similva_diagonal_),dim=1))
                loss = self.celoss(pi, x['label']) + self.celoss(ci, x['label']) + loss_similva_diagonal + self.celoss(prob, x['label'])
                return loss, prob, x['label'], self.celoss(pi, x['label']),0, self.celoss(ci,x['label']), 0, loss_similva_diagonal, 0, self.celoss(prob, x['label']), attn_maps_lva, 0, 0
            if self.cfg.ablation == 'ImgText_ST_cls_baseine' or self.cfg.ablation == 'ImgText_ST_cls_proposed':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:,384:])
                prob = self.img_encoder.encoder.fc_ImgText_ST(torch.cat((p_img_feat_g, p_text_emb_g[:,384:]), dim=1))
                loss = self.celoss(pi, x['label']) + self.celoss(pt, x['label'])+ self.celoss(prob, x['label'])
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), 0, 0, 0, 0, self.celoss(prob, x['label']), 0,0,0
            if self.cfg.ablation == 'ImgText_MT_cls_baseline':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])
                prob = self.img_encoder.encoder.fc_ImgText_MT_b(torch.cat((p_img_feat_g,p_text_emb_g[:, 384:],c_img_feat_g,text_emb_g[:, 384:]), dim=1))
                loss = self.celoss(pi, x['label']) + self.celoss(pt, x['label']) +self.celoss(ci, x['label']) + self.celoss(ct, x['label']) + self.celoss(prob, x['label'])
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci,x['label']), self.celoss(ct, x['label']), 0, 0, self.celoss(prob, x['label']), attn_maps_tem, 0, 0
            if self.cfg.ablation == 'ImgText_MT_cls_proposed': #
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])
                lva_i = self.img_encoder.encoder.fc_lva(similarities_lva)
                lta_t = self.img_encoder.encoder.fc_lta(similarities_lta)
                prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g, p_text_emb_g[:, 384:], c_img_feat_g, text_emb_g[:, 384:], similarities_lva, similarities_lta), dim=1))
                # similarities_lva,
                loss = self.celoss(pi, x['label']) + self.celoss(pt, x['label']) + self.celoss(ci, x['label']) + self.celoss(ct, x['label'])  + self.celoss(lta_t, x['label']) + self.celoss(prob, x['label'])
                # + self.celoss(lva_i, x['label'])
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci, x['label']), self.celoss(ct, x['label']), self.celoss(lva_i, x['label']), self.celoss(lta_t, x['label']), self.celoss(prob, x['label']), attn_maps_tem, lva_i, lta_t
            if self.cfg.ablation == 'ImgText_MT_pre_cls_proposed_simibceispy2':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])
                index_negativeone = torch.nonzero(x['label'] == -1).squeeze()
                similva_diagonal = torch.zeros((similarities_lva.shape[0] - len(index_negativeone), 1)).cuda()
                similva_diagonal_ce = torch.zeros((similarities_lva.shape[0], 1)).cuda()
                bce_label = torch.zeros((similarities_lva.shape[0] - len(index_negativeone), 1)).cuda()
                j = 0
                for i in range(similarities_lva.shape[0]):
                    if i not in index_negativeone:
                        similva_diagonal[j, :] = similarities_lva[i][i]

                        bce_label[j, :] = x['label'][i]
                        j = j + 1
                    similva_diagonal_ce[i, :] = similarities_lva[i][i]

                if len(index_negativeone) == similarities_lva.shape[0]:
                    loss_similva_diagonal = torch.zeros(1).cuda()

                else:
                    labels = torch.ones(bce_label.shape).cuda()
                    loss_similva_diagonal = self.bceloss(similva_diagonal, (labels - bce_label).float())

                similva_diagonal_ = []
                for count in range(16):
                    similva_diagonal_.append(similva_diagonal)
                similva_diagonal_ = torch.stack(similva_diagonal_)
                similva_diagonal_ = similva_diagonal_.view(similva_diagonal.size(0), 16)
                prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g, p_text_emb_g[:, 384:],
                                                                           c_img_feat_g, text_emb_g[:, 384:],
                                                                           similva_diagonal_),
                                                                          dim=1))

                cls_loss = torch.zeros(1).cuda()
                index = 0
                for itemloss_cls in [self.celoss(pi, x['label']), self.celoss(pt, x['label']),
                                     self.celoss(ci, x['label']), self.celoss(ct, x['label']),
                                     loss_similva_diagonal, self.celoss(prob, x['label']),
                                     ]:
                    index = index + 1
                    if (index == 5) and (itemloss_cls == torch.zeros(loss_similva_diagonal.shape).cuda()):
                        continue
                    if (not torch.isnan(itemloss_cls).any()):
                        cls_loss += itemloss_cls
                loss = cls_loss
                # + pre_loss
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci,x['label']), \
                       self.celoss(ct, x['label']), \
                       5 * loss_similva_diagonal, 5 * loss_similva_diagonal, \
                       self.celoss(prob, x['label']), attn_maps_tem, similva_diagonal_ce.detach(), similva_diagonal_ce.detach(), loss_similva_diagonal, loss_similva_diagonal
            if self.cfg.ablation == 'ImgText_MT_pre_cls_proposed_simibce':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])

                index_negativeone = torch.nonzero(x['label'] == -1).squeeze()
                similva_diagonal = torch.zeros((similarities_lva.shape[0]-len(index_negativeone), 1)).cuda()
                similta_diagonal = torch.zeros((similarities_lva.shape[0]-len(index_negativeone), 1)).cuda()
                similva_diagonal_ce = torch.zeros((similarities_lva.shape[0], 1)).cuda()
                similta_diagonal_ce = torch.zeros((similarities_lva.shape[0], 1)).cuda()
                bce_label = torch.zeros((similarities_lva.shape[0]-len(index_negativeone),1)).cuda()
                j=0
                for i in range(similarities_lva.shape[0]):
                    if i not in index_negativeone:
                        similva_diagonal[j,:] = similarities_lva[i][i]
                        similta_diagonal[j,:] = similarities_lta[i][i]
                        bce_label[j,:]=x['label'][i]
                        j=j+1
                    similva_diagonal_ce[i,:]=similarities_lva[i][i]
                    similta_diagonal_ce[i,:]=similarities_lta[i][i]

                if len(index_negativeone) ==similarities_lva.shape[0]:
                    loss_similva_diagonal = torch.zeros(1).cuda()
                    loss_similta_diagonal = torch.zeros(1).cuda()
                else:

                    labels = torch.ones(bce_label.shape).cuda()
                    loss_similva_diagonal = self.bceloss(similva_diagonal, (labels-bce_label).float())
                    loss_similta_diagonal = self.bceloss(similta_diagonal, (labels-bce_label).float())

                similva_diagonal_ = []
                similta_diagonal_ = []
                for count in range(16):
                    similva_diagonal_.append(similva_diagonal)
                    similta_diagonal_.append(similta_diagonal)
                similva_diagonal_ = torch.stack(similva_diagonal_)
                similta_diagonal_ = torch.stack(similta_diagonal_)
                similva_diagonal_ = similva_diagonal_.view(similva_diagonal.size(0),16)
                similta_diagonal_ = similta_diagonal_.view(similta_diagonal_.size(0), 16)
                prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g, p_text_emb_g[:, 384:],
                                                                           c_img_feat_g, text_emb_g[:, 384:],
                                                                           similva_diagonal_, similta_diagonal_),
                                                                           dim=1)) #new1 0.81

                cls_loss = torch.zeros(1).cuda()
                index =0
                for itemloss_cls in [self.celoss(pi, x['label']), self.celoss(pt, x['label']),
                                     self.celoss(ci, x['label']), self.celoss(ct, x['label']),
                                     loss_similva_diagonal, self.celoss(prob, x['label']),
                                     loss_similta_diagonal]:
                    index = index+1
                    if (index ==5 or index ==7) and (itemloss_cls ==  torch.zeros(loss_similva_diagonal.shape).cuda()):
                        continue
                    if (not torch.isnan(itemloss_cls).any()) :
                        cls_loss += itemloss_cls

                loss = cls_loss
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci, x['label']), self.celoss(ct, x['label']), \
                       5*loss_similva_diagonal, 5*loss_similta_diagonal, \
                       self.celoss(prob,x['label']), attn_maps_tem, similva_diagonal_ce.detach(), similta_diagonal_ce.detach(), loss_similva_diagonal, loss_similta_diagonal

            if self.cfg.ablation == 'ImgText_MT_pre_cls_proposed_simi':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])

                lva_i = self.img_encoder.encoder.fc_lva(similarities_lva)
                lta_t = self.img_encoder.encoder.fc_lta(similarities_lta)
                tem_i = self.img_encoder.encoder.fc_tem_v(t_img_feat_g)
                tem_t = self.img_encoder.encoder.fc_tem_t(text_emb_g[:, 0:384])
                if (not torch.isnan(lva_i).any()) and  (not torch.isnan(lta_t).any()) :
                    prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g,
                                                                               p_text_emb_g[:, 384:],
                                                                               c_img_feat_g,
                                                                               text_emb_g[:, 384:],
                                                                               similarities_lva,
                                                                               similarities_lta), dim=1))
                else:
                    prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g, p_text_emb_g[:, 384:],
                                                                               c_img_feat_g, text_emb_g[:, 384:],
                                                                               similarities_lta, similarities_lta),
                                                                              dim=1))
                cls_loss = torch.zeros(1).cuda()
                for itemloss_cls in [self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci, x['label']), self.celoss(ct, x['label']),
                                     self.celoss(lta_t, x['label']), self.celoss(prob, x['label']), self.celoss(lva_i, x['label'])]:
                    if not torch.isnan(itemloss_cls).any():
                        cls_loss  += itemloss_cls

                pre_loss = torch.zeros(1).cuda()

                for itemloss_pre in [lvta_loss_l, lva_lta_g_loss,lvta_loss_g, lva_lta_l_loss]:
                    if not torch.isnan(itemloss_pre).any():
                        pre_loss += itemloss_pre

                loss = cls_loss + pre_loss
                       # + pre_loss
                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci,x['label']), self.celoss(
                       ct, x['label']), self.celoss(lva_i, x['label']), self.celoss(lta_t, x['label']), self.celoss(prob,x['label']), attn_maps_tem, lva_i, lta_t, \
                       lvta_loss_l,  lva_lta_g_loss
            if self.cfg.ablation == 'ImgText_MT_pre_cls_proposed_tem':
                pi = self.img_encoder.encoder.fc_img_p(p_img_feat_g)
                pt = self.img_encoder.encoder.fc_report_p(p_text_emb_g[:, 384:])
                ci = self.img_encoder.encoder.fc_img_c(c_img_feat_g)
                ct = self.img_encoder.encoder.fc_report_c(text_emb_g[:, 384:])
                lva_i = self.img_encoder.encoder.fc_lva(similarities_lva)
                lta_t = self.img_encoder.encoder.fc_lta(similarities_lta)
                tem_i = self.img_encoder.encoder.fc_tem_v(t_img_feat_g)
                tem_t = self.img_encoder.encoder.fc_tem_t(text_emb_g[:, 0:384])
                prob = self.img_encoder.encoder.fc_ImgText_MT_p(torch.cat((p_img_feat_g, p_text_emb_g[:, 384:],
                                                                           c_img_feat_g, text_emb_g[:, 384:],
                                                                           t_img_feat_g, text_emb_g[:, 0:384]), dim=1))
                cls_loss = 0
                for itemloss_cls in [self.celoss(pi, x['label']), self.celoss(pt, x['label']),
                                     self.celoss(ci, x['label']), self.celoss(ct, x['label']),
                                     self.celoss(tem_i, x['label']), 5 * self.celoss(prob, x['label']),
                                     self.celoss(tem_t, x['label'])]:
                    if not torch.isnan(itemloss_cls).any():
                        cls_loss += itemloss_cls

                loss = cls_loss

                return loss, prob, x['label'], self.celoss(pi, x['label']), self.celoss(pt, x['label']), self.celoss(ci,x['label']), self.celoss(
                    ct, x['label']), self.celoss(lva_i, x['label']), self.celoss(lta_t, x['label']), self.celoss(prob,x['label']), attn_maps_tem, tem_i.detach(), tem_t.detach(), \
                       self.celoss(tem_i, x['label']), self.celoss(tem_t, x['label'])
            if self.cfg.ablation == 'cls_baseline':# transformer_difference--journal
                logits_d_t3 = torch.mean(
                    torch.cat((f_lcc_logits_d_t3.view(f_rmlo_logits_d_t3.shape[0], f_rmlo_logits_d_t3.shape[1], 1),
                               f_lmlo_logits_d_t3.view(f_rmlo_logits_d_t3.shape[0], f_rmlo_logits_d_t3.shape[1], 1),
                               f_rcc_logits_d_t3.view(f_rmlo_logits_d_t3.shape[0], f_rmlo_logits_d_t3.shape[1], 1),
                               f_rmlo_logits_d_t3.view(f_rmlo_logits_d_t3.shape[0], f_rmlo_logits_d_t3.shape[1], 1)),
                              dim=2), dim=2)
                logits_cp_t3 = torch.mean(torch.cat((f_lcc_logits_c_t3[self.batch_size:, :].view(
                                                    f_lcc_logits_c_t3.shape[0], f_lcc_logits_c_t3.shape[1], 1),
                                                     f_lmlo_logits_c_t3[self.batch_size:, :].view(
                                                         f_rmlo_logits_c_t3.shape[0], f_lcc_logits_c_t3.shape[1], 1),
                                                     f_rcc_logits_c_t3[self.batch_size:, :].view(
                                                         f_rmlo_logits_d_t3.shape[0], f_lcc_logits_c_t3.shape[1], 1),
                                                     f_rmlo_logits_c_t3[self.batch_size:, :].view(
                                                         f_lcc_logits_c_t3.shape[0], f_lcc_logits_c_t3.shape[1], 1)),
                                                    dim=2), dim=2)
                logits_cad_t3 = torch.mean(
                    torch.cat((f_lcc_logits_cad_t3.view(f_rmlo_logits_cad_t3.shape[0], f_rmlo_logits_cad_t3.shape[1], 1),
                               f_lmlo_logits_cad_t3.view(f_rmlo_logits_cad_t3.shape[0], f_rmlo_logits_cad_t3.shape[1], 1),
                               f_rcc_logits_cad_t3.view(f_rmlo_logits_cad_t3.shape[0], f_rmlo_logits_cad_t3.shape[1], 1),
                               f_rmlo_logits_cad_t3.view(f_rmlo_logits_cad_t3.shape[0], f_rmlo_logits_cad_t3.shape[1], 1)),
                              dim=2), dim=2)
                clstask3 = self.celoss(logits_cad_t3, x['label']) + self.celoss(logits_cp_t3, x['label']) + self.celoss(logits_d_t3, x['label'])
                loss =  clstask3
                return loss, attn_maps_tem, loss_cad_i_r, loss_temp_i_r, loss_p_i_r, loss_c_i_r, loss_current_rec, loss_previous_rec, clstask1, clstask2

    def _calc_inner_g_loss(self, i, t1, t2, b):
        inner_g  = self.inner_loss_g( i, t1, t2, b)
        return inner_g

    def _calc_inner_l_loss(self,  i, t1, t2, b):
        inner_l  = self.inner_loss_l( i, t1, t2, b)
        return inner_l

    def _calc_dh_loss(self, m, y, images, ato):
        dh_loss = self.dh_loss(m, y, images, ato)
        return dh_loss

    def _calc_blur_m_loss(self, m, y, images):
        blur_loss = self.blur_loss(m, y, images)
        return blur_loss

    def forward(self, x):
        # img encoder branch
        text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2 = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"],
            x["p_caption_ids"], x["p_attention_mask"], x["p_token_type_ids"],
        )
        t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, img_feat_l_x1, img_feat_l_x2, img_feat_l_x3, f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1, f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2, f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2, f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1, f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1, f_rmlo_logits_cad_t2,   f_lcc_logits_c_t3, f_lmlo_logits_c_t3, f_rcc_logits_c_t3, f_rmlo_logits_c_t3, f_lcc_logits_diff_t3, f_lmlo_logits_diff_t3, f_rcc_logits_diff_t3, f_rmlo_logits_diff_t3, f_lcc_logits_cad_t3, f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3  = self.image_encoder_forward(x)
        swap_textembedding, p_swap_textembedding = 0, 0

        rec_current = None
        rec_previous = None

        return  t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2,  p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, swap_textembedding, p_swap_textembedding, rec_current, rec_previous, f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1, f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2, f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2, f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1, f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1, f_rmlo_logits_cad_t2,   f_lcc_logits_c_t3, f_lmlo_logits_c_t3, f_rcc_logits_c_t3, f_rmlo_logits_c_t3, f_lcc_logits_diff_t3, f_lmlo_logits_diff_t3, f_rcc_logits_diff_t3, f_rmlo_logits_diff_t3, f_lcc_logits_cad_t3, f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3
