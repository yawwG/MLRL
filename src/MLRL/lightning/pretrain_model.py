import torch
import json
import pandas as pd
from numpy import *
from .. import builder
from .. import utils
from torch.nn import Softmax
from pytorch_lightning.core import LightningModule
from ..vlp import ImageTextInferenceEngine
from torch import nn
import torch.nn.functional as F
import csv
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
image_text_inference = ImageTextInferenceEngine(
)

class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.mlrl = builder.build_mlrl_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        # self.count = 0
        self.dm = None
        self.best_epoch = None
        self.IOU = []
        self.total_contrast_mse = 0
        self.auc = []
        self.counts = np.zeros(1)
        self.best_epoch = 0
        self.similarities = []
        self.local_similarities = []
        self.global_similarities = []
        self.attentionlinear = nn.Linear(256,1)
        self.softmax = Softmax(dim=0)
        self.softmax2 = Softmax(dim=2)
        self.dense = nn.Linear(768, 200)
        self.best_auc = 0.5
        self.all_risk_probabilities = []
        self.all_risk_label = []
        self.all_followups = []
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.predict_mtp = nn.Sequential(
            nn.Linear(384 * 3 , 6),
            # nn.Dropout(),
            nn.ReLU(),
        )  # output layer
        self.predict_ImgText_MT = nn.Linear(384 * 4 + 16 * 2, 2)
        self.predict_ImgText_MT2 = nn.Linear(384 * 4, 2)
        self.label=[]
        self.prob=[]
        self.simi_lva = []
        self.simi_lta = []
        self.z1_raw = []
        self.z2_raw = []
        self.features_g = []
        self.tfeatures_g = []
        try:
            ckpt = torch.load(cfg.lightning.ckpt)
        except:
            print('strat pretrianing')
        else:
            model_ckpt = self.mlrl.state_dict()
            ckpt_dict = ckpt["state_dict"]
            fixed_ckpt_dict = {}
            for k, v in ckpt_dict.items():

                new_key = k.split("mlrl.")[-1]
                fixed_ckpt_dict[new_key] = v

            ckpt_dict = fixed_ckpt_dict
            self.mlrl.load_state_dict(ckpt_dict)

    def Find_Optimal_Cutoff(self, TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold, point

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.mlrl)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):

        if 'mldrl' in self.cfg.ablation:
            loss = self.shared_step(batch, "val", batch_idx)
        else:
            t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, text_emb_l, text_emb_g, loss, attn_maps, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, rec_current, rec_previous = self.shared_step(
                batch, "val", batch_idx)

        return loss

    def tsne_method(self, feature_bank, feature_labels, epoch, flag):

        memory_feature_bank = feature_bank
        memory_feature_labels = feature_labels
        # memory_feature_bank = memory_feature_bank.cpu()
        # memory_feature_labels = memory_feature_labels.cpu()
        i_memory_feature_bank = []
        i_memory_feature_labels = []
        for orderi in range(0, len(memory_feature_bank)):
            i_memory_feature_bank.append(memory_feature_bank[orderi].tolist())
            i_memory_feature_labels.append(memory_feature_labels[orderi].tolist())

        memory_feature_bank = np.array(i_memory_feature_bank)
        print(memory_feature_bank.shape[0],memory_feature_bank.shape[1])
        memory_feature_bank = memory_feature_bank.reshape(-1,memory_feature_bank.shape[1])
        print(memory_feature_bank.shape)
        memory_feature_labels = np.array(i_memory_feature_labels)
        memory_feature_labels = memory_feature_labels.reshape(-1, 1)

        # t-sne
        tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
        result_2D = tsne_2D.fit_transform(memory_feature_bank)
        classesoflabes = np.unique(memory_feature_labels)

        color = ['red', 'blue', 'purple', 'green', 'yellow']
        marker = ['^', 'o', 'x', '*', 's']

        plt.figure()
        for i in range(0, len(classesoflabes)):
            result_sub = result_2D[np.where(memory_feature_labels == classesoflabes[i])[0], :]
            plt.scatter(result_sub[:, 0], result_sub[:, 1], color=color[i], marker=marker[i], alpha=0.5,
                        label=str(classesoflabes[i]))

        plt.title('t-SNE' + '-epoch' + str(epoch))
        plt.legend(loc='upper right')
        if flag == "g":
            plt.savefig(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_global_acr.png')
            print(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_global_ar.png')
        else:
            plt.savefig(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_local_acr.png')

        plt.close()

    def validation_step(self, batch, batch_idx):

        t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l,  text_emb_l, text_emb_g, loss, attn_maps, sents,text_mlp1, text_mlp2,  p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, rec_current, rec_previous = self.shared_step(batch, "val", batch_idx)

        if True:
            if self.cfg.train.update_interval is not None:
                if batch_idx % self.cfg.train.update_interval == 0:
                    i = 0
                    index = []
                    while i < c_img_feat_l.size(0):
                        index.append(i)
                        i = i + 4
                    index = np.stack(index)
                    # index = np.squeeze(index, dim=1)
                    np.random.shuffle(index)
                    index = index[0]
                    if (self.cfg.modality == 'mri') and ('MT' in self.cfg.ablation):
                        for i in range(index, index + 1):
                            imgs = batch["imgs"]
                            p_imgs = batch["p_imgs"]
                            try:
                                os.mkdir(self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-i:lva)/')
                            except:
                                print('')
                            if batch['p_path'][i] != str(-1):
                                # plot temporal features projection between previous images and current images
                                c_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                    image_path=c_img_feat_l[i].permute(1, 2, 0),  # [w,h,c]
                                    query_text=p_img_feat_g[i:i + 1, 0:384],  # [1,c]
                                    crop_size=imgs.shape[2],
                                    method='similarity-based',
                                    interpolation="bilinear",
                                )
                                p_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                    image_path=p_img_feat_l[i].permute(1, 2, 0),  # [w,h,c]
                                    query_text=c_img_feat_g[i:i + 1, 0:384],  # [1,c]
                                    crop_size=imgs.shape[2],
                                    method='similarity-based',
                                    interpolation="bilinear",
                                )
                                plot_phrase_grounding_similarity_map(
                                    image_paths=(p_imgs[i][0], imgs[i][0]),  # [w,h]
                                    similarity_maps=(p_similarity_map, c_similarity_map),
                                    paths=(self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-i:lva)/' +
                                           batch['p_path'][i] + '_val_epoch' + str(self.current_epoch) + str(
                                        batch['p_label'][i].cpu().numpy()) + '_prior.png',
                                           self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-i:lva)/' +
                                           batch['path'][i] + '_val_epoch' + str(self.current_epoch) + str(
                                               batch['label'][i].cpu().numpy()) + '.png'),
                                    title1s=('Img_prior', 'Img_current'),
                                    title2s=('Temporal_isolines', 'Temporal_isolines'),
                                    title3s=('Temporal_heatmap', 'Temporal_heatmap'),
                                    bboxess=(bboxes, bboxes)
                                )
                                if True:
                                    with torch.set_grad_enabled(True):
                                        # target_layers = [nn.Sequential(*list(self.mlrl.image_encoder.encoder.encoder.children()))[7][-1]]
                                        target_layers = [self.mlrl.img_encoder.encoder.encoder.layer4[-1]]
                                        cam = GradCAM(model=self.mlrl.img_encoder.encoder,
                                                      target_layers=target_layers, use_cuda=True)
                                        input = torch.cat((batch['imgslcc'], batch['p_imgslcc']), dim=0)
                                        grayscale_cam1_1 = cam(input_tensor=input, target_category=0)  # [b*2,512,512]
                                        # path = batch['path'] + batch['p_path']
                                        try:
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-id:vit)/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-td:clip-dynamic)/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'temproal-vision-to-report/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'cam-inference-similarity_map/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'cam-attention-map/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'attention_map/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'cam-tracking-from-vision-to-report/')
                                            os.mkdir(self.cfg.lightning.logger.save_dir + 'cam-tracking-from-vision-to-temVision/')
                                        except:
                                            print('')
                                        for i in range(grayscale_cam1_1.shape[0]):
                                            if i < (grayscale_cam1_1.shape[0] / 2):
                                                # name = batch['p_path'][i * 4] +'_p' + batch['path'][i * 4] + '_c' + '_test_' + str(batch['label'][i * 2].cpu().numpy()) + '_attention.png',
                                                # original_image = utils.torch_to_np(torch.split(batch["imgslcc"], 1, dim=0)[i])
                                                plot_phrase_grounding_similarity_map(
                                                    image_paths=(batch['p_imgslcc'][i][0], batch['imgslcc'][i][0]),
                                                    # [w,h]
                                                    similarity_maps=(
                                                    grayscale_cam1_1[i + 4, :, :], grayscale_cam1_1[i, :, :]),
                                                    paths=(self.cfg.lightning.logger.save_dir + 'cam-attention-map/' +
                                                           batch['p_path'][i * 4] + '_val_epoch' + str(self.current_epoch) + str(
                                                        batch['p_label'][i * 4].cpu().numpy()) + '_prior.png',
                                                           self.cfg.lightning.logger.save_dir + 'cam-attention-map/' +
                                                           batch['p_path'][i * 4] + '_p_' + batch['path'][
                                                               i * 4] + '_c_' + '_val_epoch' + str(self.current_epoch) + '_pcr' + str(
                                                               batch['label'][i * 4].cpu().numpy()) + '.png'),
                                                    title1s=('Img_prior', 'Img_current'),
                                                    title2s=('Temporal_isolines', 'Temporal_isolines'),
                                                    title3s=('Temporal_heatmap', 'Temporal_heatmap'),
                                                    bboxess=(bboxes, bboxes)
                                                )
                                                # cam-similarity map
                                                grayscale_cam1 = torch.from_numpy(grayscale_cam1_1).cuda()

                                                if ('Text' in self.cfg.ablation) or ('cls' not in self.cfg.ablation):

                                                    # plot  features projection  and dynamic report
                                                    c_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=c_img_feat_l[i*4].permute(1, 2, 0),  # [w,h,c]
                                                        query_text=text_emb_g[(i*4):(i*4 + 1), 0:384],  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",
                                                    )
                                                    p_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=p_img_feat_l[i*4].permute(1, 2, 0),  # [w,h,c]
                                                        query_text=text_emb_g[(i*4):(i*4 + 1), 0:384],  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",
                                                    )
                                                    plot_phrase_grounding_similarity_map(
                                                        image_paths=(batch['p_imgslcc'][i][0], batch['imgslcc'][i][0]),
                                                        # [w,h]
                                                        similarity_maps=(p_similarity_map, c_similarity_map),
                                                        paths=(
                                                            self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-td:clip-dynamic)/' +
                                                            batch['p_path'][i * 4] + '_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['p_label'][i * 4].cpu().numpy()) + '_prior.png',
                                                            self.cfg.lightning.logger.save_dir + 'tracking_heatmap(i-td:clip-dynamic)/' +
                                                            batch['p_path'][i * 4] + '_p_' + batch['path'][i * 4] + '_c_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['label'][i * 4].cpu().numpy()) + '.png'),
                                                        title1s=('Img_prior', 'Img_current'),
                                                        title2s=('Temporal_isolines', 'Temporal_isolines'),
                                                        title3s=('Temporal_heatmap', 'Temporal_heatmap'),
                                                        bboxess=(bboxes, bboxes)
                                                    )

                                                    # plot temporal vision features projection to report
                                                    c_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=t_img_feat_l[i*4].permute(1, 2, 0),  # [w,h,c]
                                                        query_text=text_emb_g[(i*4):(i*4 + 1), 0:384],  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",
                                                    )
                                                    p_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=t_img_feat_l[i*4].permute(1, 2, 0),  # [w,h,c]
                                                        query_text=p_text_emb_g[(i*4):(i*4 + 1), 0:384],  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",

                                                    )
                                                    plot_phrase_grounding_similarity_map(
                                                        image_paths=(batch['p_imgslcc'][i][0], batch['imgslcc'][i][0]),
                                                        # [w,h]
                                                        similarity_maps=(p_similarity_map, c_similarity_map),
                                                        paths=(
                                                            self.cfg.lightning.logger.save_dir + 'temproal-vision-to-report/' +
                                                            batch['p_path'][i*4] + '_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['p_label'][i*4].cpu().numpy()) + '_prior.png',
                                                            self.cfg.lightning.logger.save_dir + 'temproal-vision-to-report/' +
                                                            batch['p_path'][i * 4] + '_p_' + batch['path'][i * 4] + '_c_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['label'][i * 4].cpu().numpy()) + '.png'),
                                                        title1s=('Img_prior', 'Img_current'),
                                                        title2s=('Temporal_isolines', 'Temporal_isolines'),
                                                        title3s=('Temporal_heatmap', 'Temporal_heatmap'),
                                                        bboxess=(bboxes, bboxes)
                                                    )

                                                    # plot cam visual features projection to the report
                                                    text_emb_g_ = torch.max(text_emb_g[(i*4):(i*4 + 1), 0:384], dim=1)[0]
                                                    p_text_emb_g_ = torch.max(p_text_emb_g[(i*4):(i*4 + 1), 0:384], dim=1)[0]
                                                    c_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=torch.split(grayscale_cam1, 1, dim=0)[i].permute(1,
                                                                                                                    2,
                                                                                                                    0),
                                                        # [w,h,c]
                                                        query_text=text_emb_g_.view(1, 1),  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",
                                                    )
                                                    p_similarity_map = image_text_inference.get_similarity_map_from_raw_data(
                                                        image_path=torch.split(grayscale_cam1, 1, dim=0)[i + 4].permute(
                                                            1,
                                                            2,
                                                            0),
                                                        # [w,h,c]
                                                        query_text=p_text_emb_g_.view(1, 1),  # [1,c]
                                                        crop_size=batch['imgslcc'].shape[2],
                                                        method='similarity-based',
                                                        interpolation="bilinear",

                                                    )
                                                    plot_phrase_grounding_similarity_map(
                                                        image_paths=(batch['p_imgslcc'][i][0], batch['imgslcc'][i][0]),
                                                        # [w,h]
                                                        similarity_maps=(p_similarity_map, c_similarity_map),
                                                        paths=(
                                                            self.cfg.lightning.logger.save_dir + 'cam-tracking-from-vision-to-report/' +
                                                            batch['p_path'][i*4] + '_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['p_label'][i*4].cpu().numpy()) + '_prior.png',
                                                            self.cfg.lightning.logger.save_dir + 'cam-tracking-from-vision-to-report/' +
                                                            batch['p_path'][i * 4] + '_p_' + batch['path'][i * 4] + '_c_val_epoch' + str(
                                                                self.current_epoch) + str(
                                                                batch['label'][i * 4].cpu().numpy()) + '.png'),
                                                        title1s=('Img_prior', 'Img_current'),
                                                        title2s=('Temporal_isolines', 'Temporal_isolines'),
                                                        title3s=('Temporal_heatmap', 'Temporal_heatmap'),
                                                        bboxess=(bboxes, bboxes)
                                                    )

                                                    attencsv = {
                                                        'attenscores': [],
                                                        'descs': [],
                                                    }
                                                    atten_temp_word = attn_maps[i*4].half()  #[B,w*h,word_num]
                                                    atten_temp = atten_temp_word.view(atten_temp_word.shape[0],atten_temp_word.shape[1],atten_temp_word.shape[2] * atten_temp_word.shape[3])
                                                    atten_temp = torch.sum(atten_temp, dim=2) #[1,200,1]
                                                    atten_temp = (atten_temp - atten_temp.min()) / (atten_temp.max() - atten_temp.min())
                                                    atten_temp = torch.squeeze(atten_temp) #[200]
                                                    atten_temp_score_ = self.softmax(atten_temp)
                                                    for n in range(attn_maps[i*4].size(1)):
                                                        atten_temp_score = atten_temp_score_[n].detach().cpu().numpy()
                                                        attencsv['attenscores'].append(atten_temp_score)
                                                        desc = sents[i*4][n]
                                                        attencsv['descs'].append(desc)
                                                        # plot attetntion
                                                        # atten_temp_ = torch.split(atten_temp_word, 1, dim=1)[n]
                                                        # if batch['label'][i] == torch.ones(1).cuda():
                                                        #     try:
                                                        #         a2 = utils.torch_to_np(atten_temp_)
                                                        #         desc = desc.replace("/", "")
                                                        #         name = batch['path'][i].split('/')[-1].split('.png')[0] + "_atten_" + str(n) + '_' + desc + '_' + str(atten_temp_score) + '_val_epoch' + str(self.current_epoch)
                                                        #         input = utils.torch_to_np(torch.split(batch["imgs"], 1, dim=0)[i])
                                                        #         utils.save_image2(input, name, a2,self.cfg.output_dir + 'attention_map/')
                                                        #     except:
                                                        #         continue
                                                    results_data_frame = pd.DataFrame(data=attencsv)
                                                    results_data_frame.to_csv(self.cfg.output_dir + 'attention_map/' + 'atten_score' + '_val_epoch_' +
                                                        batch['p_path'][i * 4] + '_p_' + batch['path'][i * 4] + '_c_' + 'val_epoch' + str(self.current_epoch) +'_pcr' + str(
                                                            batch['label'][i * 4].cpu().numpy()) + '.csv')

        return loss

    def softmax_method(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        return x

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = np.sum(x1 * x2, dim)

        w1 = np.linalg.norm(x1, 2)
        w2 = np.linalg.norm(x2, 2)
        cosine2 = (w12 / (w1 * w2))
        # cosine = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        cosine2 = (cosine2-np.min(cosine2))/(np.max(cosine2)-np.min(cosine2))
        return np.expand_dims(cosine2,axis=1)

    def one_hot(self, x, class_count):
        return torch.eye(class_count)[x, :]

    def me_sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def shared_step(self, batch, split, batch_idx):
        """Similar to traning step"""

        if 'cls' in self.cfg.ablation:
            if 'pre' in self.cfg.ablation:# pre+cls end-to-end
                loss, prob, label, pi_loss, pt_loss, ci_loss, ct_loss, lva_i_loss, lta_t_loss, cls_loss, attn_maps_lva, similarities_lva, similarities_lta, tem_i_loss, tem_t_loss = self.mlrl.calc_loss(
                    split,
                    t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g,
                    cad_img_feat_l, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g,
                    p_sents, p_text_mlp1, p_text_mlp2, swap_textembedding, p_swap_textembedding, rec_current, rec_previous,
                    f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1,
                    f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2,
                    f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2,
                    f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1,
                    f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1,
                    f_rmlo_logits_cad_t2, f_lcc_logits_c_t3, f_lmlo_logits_c_t3, f_rcc_logits_c_t3, f_rmlo_logits_c_t3,
                    f_lcc_logits_d_t3, f_lmlo_logits_d_t3, f_rcc_logits_d_t3, f_rmlo_logits_d_t3, f_lcc_logits_cad_t3,
                    f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3, batch, self.current_epoch
                    )
                self.log(
                    f"{split}_tem_i_loss",
                    tem_i_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

                self.log(
                    f"{split}_tem_t_loss",
                    tem_t_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )
                # if split!='train':
                self.log(
                    f"{split}_pi_loss",
                    pi_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

                self.log(
                    f"{split}_pt_loss",
                    pt_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

                self.log(
                    f"{split}_ci_loss",
                    ci_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    f"{split}_ct_loss",
                    ct_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

                self.log(
                    f"{split}_lva_i_loss",
                    lva_i_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

                self.log(
                    f"{split}_lta_t_loss",
                    lta_t_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    f"{split}_cls_loss",
                    cls_loss,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

            log_iter_loss = True if split == "train" else False

            self.log(
                f"{split}_loss",
                loss,
                on_epoch=True,
                on_step=log_iter_loss,
                logger=True,
                prog_bar=True,
            )
            try:
                pi_index=0
                if (batch_idx == 0 and split != 'train'):
                    self.label = []
                    self.prob = []
                    self.simi_lva = []
                    self.simi_lta = []

                for la, pr, pr_lva, pr_lta in zip(label, prob, similarities_lva, similarities_lta):

                    if la == torch.zeros(1).cuda() or la == torch.ones(1).cuda():
                        self.label.append(la)
                        self.prob.append(pr.float())
                        self.z1_raw.append(torch.split(p_img_feat_g,1,dim=0)[pi_index])
                        self.z2_raw.append(torch.split(c_img_feat_g,1,dim=0)[pi_index])
                        if 'ImgText_MT_pre_cls_proposed_simibce' == self.cfg.ablation:
                            self.simi_lva.append(1-self.me_sigmoid(pr_lva.float().detach().cpu().numpy()))
                            self.simi_lta.append(1-self.me_sigmoid(pr_lta.float().detach().cpu().numpy()))
                        else:
                            self.simi_lva.append(self.me_sigmoid(pr_lva.float().detach().cpu().numpy()))
                            self.simi_lta.append(self.me_sigmoid(pr_lta.float().detach().cpu().numpy()))
                    pi_index = pi_index + 1
            except:
                for la, pr in zip(label, prob):
                    if la == torch.zeros(1).cuda() or la == torch.ones(1).cuda():
                        self.label.append(la)
                        self.prob.append(pr.float())

            self.features_g.append(c_img_feat_g)
            self.tfeatures_g.append(p_img_feat_g)

            if (batch_idx == 2) :

                y = torch.stack(self.label)
                # y = self.one_hot(y,2).long()
                y = y.long()
                y = y.detach().cpu().numpy()

                if True: #tsne
                    self.features_g = torch.stack(self.features_g)
                    self.features_g = self.features_g.view(self.features_g.size(0) * self.features_g.size(1), self.features_g.size(2))
                    self.tfeatures_g = torch.stack(self.tfeatures_g)
                    self.tfeatures_g = self.tfeatures_g.view(self.tfeatures_g.size(0) * self.tfeatures_g.size(1), self.tfeatures_g.size(2))

                    gfeature_bank = torch.cat((self.features_g, self.tfeatures_g), 0)
                    # gfeature_bank = gfeature_bank.view(gfeature_bank.size(0) * gfeature_bank.size(1), gfeature_bank.size(2))
                    feature_labels = torch.cat((torch.zeros(self.features_g.shape[0],1),torch.ones(self.features_g.shape[0],1)), dim=0)
                    self.tsne_method(gfeature_bank, feature_labels, 'test', "g")

                prob = torch.stack(self.prob)

                prob = prob.detach().cpu().numpy()
                try:
                    simi_lva = np.stack(self.simi_lva)
                    simi_lta = np.stack(self.simi_lta)
                    # simi_lva = o_simi_lva
                    # simi_lta = o_simi_lta
                except:
                    simi_lva = self.simi_lva
                    simi_lta = self.simi_lta
                    try:
                        testshape = simi_lva.shape[1]
                    except:
                        simi_lva = prob
                        simi_lta = prob
                auroc_list, auroc_list_lva, auroc_list_lta, auprc_list = [], [],  [], []
                for i in range(1):
                    # print(y.shape)
                    # print(prob.shape)
                    y_cls = y[:]
                    prob_cls = prob[:,1]
                    if simi_lva.shape[1]==2:
                        prob_lva = simi_lva[:,1]
                        prob_lta = simi_lta[:,1]
                    else:
                        try:
                            prob_lva = simi_lva[:]
                            prob_lta = simi_lta[:]
                        except:
                            print('')
                    if np.isnan(prob_cls).any():
                        auprc_list.append(0)
                        auroc_list.append(0)
                    else:
                        try:
                            auroc_list.append(roc_auc_score(y_cls, prob_cls))
                            try:
                                auroc_list_lta.append(roc_auc_score(y_cls, prob_lta))
                                auroc_list_lva.append(roc_auc_score(y_cls, prob_lva))
                            except:
                                print('')
                        except:
                            print('problem!,prob_cls:', prob_cls)
                            # auroc_list.append(0)

                auroc = np.mean(auroc_list)
                prob_ = np.where(prob > 0.5, 1, 0)
                try:
                    auroc_lva = np.mean(auroc_list_lva)
                    auroc_lta = np.mean(auroc_list_lta)
                    self.log(f"{split}_auroc_lva", auroc_lva, on_epoch=True, logger=True, prog_bar=True)
                    self.log(f"{split}_auroc_lta", auroc_lta, on_epoch=True, logger=True, prog_bar=True)

                except:
                    auroc_lva = 0
                    auroc_lta = 0
                self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)

                self.label = []
                self.prob = []
                self.simi_lva = []
                self.simi_lta = []
                self.features_g = []
                self.tfeatures_g = []
            if 'mldrl' not in self.cfg.ablation:
                return t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l,  text_emb_l, text_emb_g, loss, attn_maps_lva, sents, text_mlp1, text_mlp2,  p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, rec_current, rec_previous
            else:
                return loss
        else: #pretrain
            try: # mri
                loss, attn_maps, loss_cad_i_r ,loss_temp_i_r, loss_p_i_r, loss_c_i_r, loss_current_rec, loss_previous_rec, clstask1, clstask2,  similarities_lva, similarities_lta, loss_lvta, loss_lva_lta = self.mlrl.calc_loss(split,
                    t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g,
                    cad_img_feat_l, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g, p_sents,
                    p_text_mlp1, p_text_mlp2,  swap_textembedding, p_swap_textembedding, rec_current, rec_previous, f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1, f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2, f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2, f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1, f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1, f_rmlo_logits_cad_t2, f_lcc_logits_c_t3,  f_lmlo_logits_c_t3,  f_rcc_logits_c_t3,  f_rmlo_logits_c_t3, f_lcc_logits_d_t3, f_lmlo_logits_d_t3, f_rcc_logits_d_t3, f_rmlo_logits_d_t3, f_lcc_logits_cad_t3, f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3, batch, self.current_epoch
                )
                self.log(
                    f"{split}_loss_lvta",
                    loss_lvta,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    f"{split}_loss_lta",
                    loss_lva_lta,
                    on_epoch=True,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

            except:
                loss, attn_maps, loss_cad_i_r, loss_temp_i_r, loss_p_i_r, loss_c_i_r, loss_current_rec, loss_previous_rec, clstask1, clstask2 = self.mlrl.calc_loss(
                    split,
                    t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g,
                    cad_img_feat_l, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g,
                    p_sents,
                    p_text_mlp1, p_text_mlp2, swap_textembedding, p_swap_textembedding, rec_current, rec_previous,
                    f_lcc_logits_cp_t1, f_lcc_logits_d_t1, f_lmlo_logits_cp_t1, f_lmlo_logits_d_t1, f_rcc_logits_cp_t1,
                    f_rcc_logits_d_t1, f_rmlo_logits_cp_t1, f_rmlo_logits_d_t1, f_lcc_logits_cp_t2, f_lcc_logits_d_t2,
                    f_lmlo_logits_cp_t2, f_lmlo_logits_d_t2, f_rcc_logits_cp_t2, f_rcc_logits_d_t2, f_rmlo_logits_cp_t2,
                    f_rmlo_logits_d_t2, f_lcc_logits_cad_t1, f_lcc_logits_cad_t2, f_lmlo_logits_cad_t1,
                    f_lmlo_logits_cad_t2, f_rcc_logits_cad_t1, f_rcc_logits_cad_t2, f_rmlo_logits_cad_t1,
                    f_rmlo_logits_cad_t2, f_lcc_logits_c_t3, f_lmlo_logits_c_t3, f_rcc_logits_c_t3, f_rmlo_logits_c_t3,
                    f_lcc_logits_d_t3, f_lmlo_logits_d_t3, f_rcc_logits_d_t3, f_rmlo_logits_d_t3, f_lcc_logits_cad_t3,
                    f_lmlo_logits_cad_t3, f_rcc_logits_cad_t3, f_rmlo_logits_cad_t3, batch, self.current_epoch
                    )

            if self.cfg.modality == 'mri':
                if (split != 'train' and batch_idx == 0):
                    self.label = []
                    self.prob = []
                    self.simi_lva = []
                    self.simi_lta = []
                try:
                    x_p_c_d = torch.cat((p_img_feat_g, p_text_emb_g[:, 384:], c_img_feat_g, text_emb_g[:, 384:], similarities_lva, similarities_lta), dim=1)
                    prob = self.predict_ImgText_MT(x_p_c_d)
                except:
                    x_p_c_d = torch.cat((p_img_feat_g, p_text_emb_g[:, 384:], c_img_feat_g, text_emb_g[:, 384:]), dim=1)
                    prob = self.predict_ImgText_MT2(x_p_c_d)
                for la, pr in zip(batch['label'], prob):
                    if la == torch.zeros(1).cuda() or la == torch.ones(1).cuda():
                        self.label.append(la.detach())
                        self.prob.append(pr.detach().float())

                if (split != 'train' and batch_idx == batch_idx_test) or (split == 'train' and batch_idx == batch_idx_train):

                    y = torch.stack(self.label)
                    y = y.long()
                    try:
                        y = y.detach().cpu().numpy()
                    except:
                        y = y.cpu().numpy()
                    prob = torch.stack(self.prob)
                    try:
                        prob = prob.detach().cpu().numpy()
                    except:
                        prob = prob.cpu().numpy()
                    try:
                        simi_lva = np.stack(self.simi_lva)
                        simi_lta = np.stack(self.simi_lta)
                    except:
                        simi_lva = self.simi_lva
                        simi_lta = self.simi_lta
                        try:
                            testshape = simi_lva.shape[1]
                        except:
                            simi_lva = prob
                            simi_lta = prob
                    auroc_list, auroc_list_lva, auroc_list_lta, auprc_list = [], [], [], []
                    for i in range(1):
                        y_cls = y[:]
                        prob_cls = prob[:, 1]
                        if simi_lva.shape[1] == 2:
                            prob_lva = simi_lva[:, 1]
                            prob_lta = simi_lta[:, 1]
                        else:
                            try:
                                prob_lva = simi_lva[:]
                                prob_lta = simi_lta[:]
                            except:
                                print('')
                        if np.isnan(prob_cls).any():
                            auprc_list.append(0)
                            auroc_list.append(0)
                        else:
                            try:
                                auroc_list.append(roc_auc_score(y_cls, prob_cls))
                                try:
                                    auroc_list_lta.append(roc_auc_score(y_cls, prob_lta))
                                    auroc_list_lva.append(roc_auc_score(y_cls, prob_lva))
                                except:
                                    print('')
                            except:
                                print('problem!,prob_cls:', prob_cls)

                    auroc = np.mean(auroc_list)
                    prob_ = np.where(prob > 0.5, 1, 0)
                    try:
                        auroc_lva = np.mean(auroc_list_lva)
                        auroc_lta = np.mean(auroc_list_lta)
                        self.log(f"{split}_auroc_lva", auroc_lva, on_epoch=True, logger=True, prog_bar=True)
                        self.log(f"{split}_auroc_lta", auroc_lta, on_epoch=True, logger=True, prog_bar=True)

                    except:
                        auroc_lva = 0
                        auroc_lta = 0
                    self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)

                    try:
                        # if True:
                        if split != "train":
                            # if True:
                            if (self.best_auc < auroc):
                                self.best_auc = auroc
                                self.best_epoch = self.current_epoch
                                with open(self.cfg.output_dir + split + 'pro.csv', 'w', newline='') as file:
                                    mywriter = csv.writer(file, delimiter=',')
                                    mywriter.writerows(prob)
                                try:
                                    if 'ImgText_MT_pre_cls_proposed_simibce' == self.cfg.ablation:
                                        with open(self.cfg.output_dir + split + 'pro_lva.csv', 'w', newline='') as file:
                                            mywriter = csv.writer(file, delimiter=',')
                                            mywriter.writerows(simi_lva)
                                        with open(self.cfg.output_dir + split + 'pro_lta.csv', 'w', newline='') as file:
                                            mywriter = csv.writer(file, delimiter=',')
                                            mywriter.writerows(simi_lta)
                                    else:
                                        with open(self.cfg.output_dir + split + 'pro_lva.csv', 'w', newline='') as file:
                                            mywriter = csv.writer(file, delimiter=',')
                                            mywriter.writerows(self.softmax_method(simi_lva))
                                        with open(self.cfg.output_dir + split + 'pro_lta.csv', 'w', newline='') as file:
                                            mywriter = csv.writer(file, delimiter=',')
                                            mywriter.writerows(self.softmax_method(simi_lta))
                                except:
                                    print('')

                            results = {"epoch": self.current_epoch, "auroc": auroc, "bestauc": self.best_auc,
                                       "auroc_lva": auroc_lva, "auroc_lta": auroc_lta,
                                       "bestEpoch": self.best_epoch}
                            results_csv = os.path.join(self.cfg.output_dir, split + "_auc.csv")
                            with open(results_csv, "a") as fp:
                                json.dump(results, fp)
                                json.dump("/n", fp)
                    except:
                        print('')
                    self.label = []
                    self.prob = []
                    self.simi_lva = []
                    self.simi_lta = []

            else:
                x_p_c_d = torch.cat((p_img_feat_l, c_img_feat_l, t_img_feat_l), dim=1)
                x_p_c_d = self.pooling(x_p_c_d)
                risk = self.predict_mtp(x_p_c_d.view(x_p_c_d.size(0), x_p_c_d.size(1)))
                risk_label = batch['label']
                years_last_followup = batch['years_to_last_followup']
                risk_label = risk_label.cuda()
                pred_risk_label = F.softmax(risk, dim=1)
                years_last_followup = years_last_followup.cuda()
                self.all_risk_probabilities.append(pred_risk_label.detach().cpu().numpy())
                self.all_risk_label.append(risk_label.cpu().numpy())
                self.all_followups.append(years_last_followup.cpu().numpy())

                metrics_, _ = utils.conput_auc_cindex(self.all_risk_probabilities, self.all_risk_label, self.all_followups,6, 5)
                for year in range(5):
                    x = int(year + 1)
                    try:
                        self.log(
                            f"{split}_riskauc_"+str(x),
                            metrics_[x],
                            on_epoch=True,
                            on_step=True,
                            logger=True,
                            prog_bar=True,
                        )
                    except:
                        self.log(
                            f"{split}_riskauc_"+str(x),
                            0,
                            on_epoch=True,
                            on_step=True,
                            logger=True,
                            prog_bar=True,
                        )
                try:
                    self.log(
                        f"{split}_risk_c_index",
                        metrics_['c_index'],
                        on_epoch=True,
                        on_step=True,
                        logger=True,
                        prog_bar=True,
                    )
                except:
                    self.log(
                        f"{split}_risk_c_index",
                        0,
                        on_epoch=True,
                        on_step=True,
                        logger=True,
                        prog_bar=True,
                    )

                # if split!='train'and batch_idx == 1576:
                if split != 'train' and batch_idx == 1576:
                        # 9588: train
                    self.all_risk_probabilities = []
                    self.all_risk_label = []
                    self.all_followups = []
                # if split=='train'and batch_idx == 9588:
                if split != 'train' and batch_idx == 1576:
                    self.all_risk_probabilities = []
                    self.all_risk_label = []
                    self.all_followups = []
            # log training progress
            log_iter_loss = True if split == "train" else False
            # if loss !=None:
            self.log(
                f"{split}_loss",
                loss,
                on_epoch=True,
                on_step=log_iter_loss,
                logger=True,
                prog_bar=True,
            )
            if split != "train":
                return t_img_feat_g, t_img_feat_l, c_img_feat_g, c_img_feat_l, p_img_feat_g, p_img_feat_l, cad_img_feat_g, cad_img_feat_l, text_emb_l, text_emb_g, loss, attn_maps, sents, text_mlp1, text_mlp2, p_text_emb_l, p_text_emb_g, p_sents, p_text_mlp1, p_text_mlp2, rec_current, rec_previous
            else:
                return loss
