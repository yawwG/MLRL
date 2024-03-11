import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
import SimpleITK as sitk
from monai.transforms import (
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)
class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor()])
        self.split = split
        self.df = pd.read_csv('/dataset.csv')
        self.df = self.df.loc[(self.df['split'] == split)]

        if cfg.data.frac != 1 and split == "train":
             self.df = self.df.sample(frac=cfg.data.frac)

        # load studies and study to text mapping
        self.filenames, self.p_filenames, self.path2sent, self.label, self.keyword = self.load_text_data(split)
        self.reports =  self.df['report']
        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.clip_min = 0
        self.clip_max = 4000
        self.orientation = Orientationd(keys=["i1", "i2"], axcodes="LPS")
        self.imgpath = '/'

    def load_nii(self, nii_file):
        itk_img = sitk.ReadImage(nii_file)
        img = sitk.GetArrayFromImage(itk_img)
        # array = img.transpose(2,1,0)  # transfer to x,y,z
        return img

    def load_text_data(self, split):

        if self.cfg.model.text.bert_type == 'emilyalsentzer/Bio_ClinicalBERT':
            filepath = os.path.join(INB_DATA_DIR, "captions_"+split +".pickle")

        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)
        filenames = []
        p_filenames = []
        for idx, row in tqdm.tqdm(self.df.iterrows(), total=self.df.shape[0]):
            filenames.append(row['image_state_2'])
            p_filenames.append(row['image_state_1'])

        label = self.df[self.df['split'] == split][
            'pcr'].tolist()

        keyword = 0

        return filenames, p_filenames, path2sent, label, keyword

    def get_caption(self, path):

        series_sents = self.path2sent[path]
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, p_img_path, transform=None):

        img = []
        rotation_degree = random.randint(-30, 30)
        random_index = random.random()
        # for i in range(len(img_path)):
        if (os.path.exists(img_path)):
            img_tmp = cv2.imread(img_path, 0)
            img_tmp = self._resize_img(img_tmp, self.cfg.data.image.imsize, rotation_degree, random.random()) #xxxx
            img_tmp = Image.fromarray(img_tmp).convert("RGB")
        else:
            img_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")
        if transform is not None:
            img_tmp = transform(img_tmp)
            img.append(img_tmp)
        img = torch.stack(img)

        p_img = []

        if (os.path.exists(p_img_path)):
            p_img_tmp = cv2.imread(p_img_path, 0)
            p_img_tmp = self._resize_img(p_img_tmp, self.cfg.data.image.imsize, rotation_degree, random.random())  # xxxx
            p_img_tmp = Image.fromarray(p_img_tmp).convert("RGB")
        else:
            p_img_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")
        if transform is not None:
            p_img_tmp = transform(p_img_tmp)
            p_img.append(p_img_tmp)
        p_img = torch.stack(p_img)
        if self.split == 'train':
            img, p_img = self.my_transform(img, p_img)
        return img, p_img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = str(row["aid"])

        key = self.filenames[index]
        p_key = self.p_filenames[index]

        date1 = str((row['image_state_1']))
        date2 = str((row['image_state_2']))
        mri_img_path = img_path
        # read mip
        if (os.path.exists(self.imgpath + p_key+ ".png")):
            x_idx1 = self.imgpath + p_key + ".png"  # sip

        if (os.path.exists(self.imgpath + key + ".png")):
            x_idx2 = self.imgpath + key + ".png"  # sip

        p_imgs, imgs  = self.get_imgs(x_idx1, x_idx2, self.transform)

        flag = 1
        # pcr label
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        p_label = torch.tensor(label, dtype=torch.long)

        # read report
        p_caps, p_cap_len = self.get_caption(p_key)
        # try:
        caps, cap_len = self.get_caption(p_key)

        label1, keyword1, label2, keyword2 = -1, 'none', -1, 'none'
        label1 = torch.tensor(label1, dtype=torch.long)
        label2 = torch.tensor(label2, dtype=torch.long)
        flag = torch.tensor(flag, dtype=torch.long)

        keyword = 0
        years_to_last_followup = torch.tensor(0, dtype=torch.long)

        return imgs, p_imgs, caps, cap_len, p_caps, p_cap_len, key, p_key, label, p_label, date1, date2, keyword, flag, years_to_last_followup, label1, keyword1, label2, keyword2

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        # translator = Translator()
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row['report']) == str:
                captions += row['report']

            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    # if len(t) > 0:
                    included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))
            path2sent[row['image_state_1']] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

    def _resize_img(self, img, scale, rotation_degree, random_index):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        # desireable_size = (scale,scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def _resize_img_deformationn(self, img, scale, rotation_degree, random_index):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """

        desireable_size = (scale,scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        if random_index > 0.5:

            height, width = resized_img.shape[:2]

            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degree, 1)

            # Apply the rotation to the image
            resized_img = cv2.warpAffine(resized_img, rotation_matrix, (width, height))

        return resized_img

def multimodal_collate_fn(batch):
    """sort sequence"""

    imgs, imgslcc, imgslmlo, imgsrcc, imgsrmlo, masks, cap_len, ids, tokens, attention, path, labels, dates, atomo, keywords, years_to_last_followups, label1s, keyword1s, label2s, keyword2s = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    p_imgs, p_imgslcc, p_imgslmlo, p_imgsrcc, p_imgsrmlo, p_cap_len, p_ids, p_tokens, p_attention, p_path, p_dates, flags, p_labels  =  [], [], [], [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        # img, msk, cap, cap_l, p, label, ato, keyword = b
        img, p_img, cap, cap_l, p_cap, p_cap_l, key, p_key, label, p_label, date, p_date,  keyword, flag, years_to_last_followup, label1, keyword1, label2, keyword2 = b
        imgs.append(img)
        p_imgs.append(p_img)
        cap_len.append(cap_l)
        p_cap_len.append(p_cap_l)
        ids.append(cap["input_ids"])
        try:
            tokens.append(cap["token_type_ids"])
        except:
            tokens.append(cap["input_ids"])
        attention.append(cap["attention_mask"])
        p_ids.append(p_cap["input_ids"])
        try:
            p_tokens.append(p_cap["token_type_ids"])
        except:
            p_tokens.append(p_cap["input_ids"])

        p_attention.append(p_cap["attention_mask"])
        # path.append(p.split("/")[-1].split("_")[0])
        path.append(key)
        p_path.append(p_key)
        labels.append(label)
        p_labels.append(p_label)
        dates.append(date)
        p_dates.append(p_date)
        label1s.append(label1)
        label2s.append(label2)
        keyword1s.append(keyword1)
        keyword2s.append(keyword2)

        flags.append(flag)
        years_to_last_followups.append(years_to_last_followup)
        keywords.append(keyword)

    # stack
    imgs = torch.stack(imgs)
    p_imgs = torch.stack(p_imgs)

    ids = torch.stack(ids).squeeze()
    # tokens = [itm for itm in tokens for i in range(4)]
    tokens = torch.stack(tokens).squeeze()
    # attention = [itm for itm in attention for i in range(4)]
    attention = torch.stack(attention).squeeze()
    # labels = [itm for itm in labels for i in range(4)]
    p_ids = torch.stack(p_ids).squeeze()
    p_tokens = torch.stack(p_tokens).squeeze()
    p_attention = torch.stack(p_attention).squeeze()
    p_label = torch.stack(p_labels)
    label = torch.stack(labels)
    label1 = torch.stack(label1s)
    label2 = torch.stack(label2s)

    date = dates
    p_date = p_dates
    flag = torch.stack(flags)
    years_to_last_followup = torch.stack(years_to_last_followups)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    p_sorted_cap_lens, p_sorted_cap_indices = torch.sort(torch.tensor(p_cap_len), 0, True)

    path_tmp = []
    p_path_tmp = []
    keyword1 = []
    keyword2 = []
    for i in range(len(path)):

        path_tmp.append(path[sorted_cap_indices[i]])
        p_path_tmp.append(p_path[sorted_cap_indices[i]])

        keyword1.append(keyword1s[sorted_cap_indices[i]])
        keyword2.append(keyword2s[sorted_cap_indices[i]])

    return_dict = {
        "imgs": torch.stack([itm for itm in imgs[sorted_cap_indices].view(len(path), 3, imgs[sorted_cap_indices].size(3),
                                              imgs[sorted_cap_indices].size(4)) for i in range(4)]),
        "p_imgs": torch.stack([itm for itm in p_imgs[p_sorted_cap_indices].view(len(path), 3, p_imgs[p_sorted_cap_indices].size(3),
                                              p_imgs[p_sorted_cap_indices].size(4)) for i in range(4)]),

        "caption_ids": torch.stack([itm for itm in ids[sorted_cap_indices] for i in range(4)]),
        "token_type_ids": torch.stack([itm for itm in tokens[sorted_cap_indices] for i in range(4)]),
        "attention_mask": torch.stack([itm for itm in attention[sorted_cap_indices] for i in range(4)]),

        "p_caption_ids": torch.stack([itm for itm in p_ids[p_sorted_cap_indices] for i in range(4)]),
        "p_token_type_ids": torch.stack([itm for itm in p_tokens[p_sorted_cap_indices] for i in range(4)]),
        "p_attention_mask": torch.stack([itm for itm in p_attention[p_sorted_cap_indices] for i in range(4)]),

        "imgslcc": imgs[sorted_cap_indices].view(len(path), 3, imgs[sorted_cap_indices].size(3),
                                                    imgs[sorted_cap_indices].size(4)),
        "imgslmlo": imgs[sorted_cap_indices].view(len(path), 3, imgs[sorted_cap_indices].size(3),
                                                      imgs[sorted_cap_indices].size(4)),
        "imgsrcc": imgs[sorted_cap_indices].view(len(path), 3, imgs[sorted_cap_indices].size(3),
                                                    imgs[sorted_cap_indices].size(4)),
        "imgsrmlo": imgs[sorted_cap_indices].view(len(path), 3, imgs[sorted_cap_indices].size(3),
                                                      imgs[sorted_cap_indices].size(4)),

        "p_imgslcc": p_imgs[p_sorted_cap_indices].view(len(p_path), 3, p_imgs[p_sorted_cap_indices].size(3),
                                                    p_imgs[p_sorted_cap_indices].size(4)),
        "p_imgslmlo": p_imgs[p_sorted_cap_indices].view(len(p_path), 3, p_imgs[p_sorted_cap_indices].size(3),
                                                      p_imgs[p_sorted_cap_indices].size(4)),
        "p_imgsrcc": p_imgs[p_sorted_cap_indices].view(len(p_path), 3, p_imgs[p_sorted_cap_indices].size(3),
                                                    p_imgs[p_sorted_cap_indices].size(4)),
        "p_imgsrmlo": p_imgs[p_sorted_cap_indices].view(len(p_path), 3, p_imgs[p_sorted_cap_indices].size(3),
                                                      p_imgs[p_sorted_cap_indices].size(4)),
        "cap_lens": [itm for itm in sorted_cap_lens for i in range(4)],

        "p_cap_lens": [itm for itm in p_sorted_cap_lens for i in range(4)],

        "path":  [itm for itm in path_tmp for i in range(4)],

        "p_path": [itm for itm in p_path_tmp for i in range(4)],

        "label": torch.stack([itm for itm in label[sorted_cap_indices] for i in range(4)]),

        "p_label": torch.stack([itm for itm in p_label[sorted_cap_indices] for i in range(4)]),

        "label1": torch.stack([itm for itm in label1[sorted_cap_indices] for i in range(1)]),

        "label2": torch.stack([itm for itm in label2[sorted_cap_indices] for i in range(1)]),

        "keyword1": [itm for itm in keyword1 for i in range(4)],

        "keyword2": [itm for itm in keyword2 for i in range(4)],

        "flag": torch.stack([itm for itm in flag[sorted_cap_indices] for i in range(4)]),

        "years_to_last_followup": torch.stack([itm for itm in years_to_last_followup[sorted_cap_indices] for i in range(4)]),

    }

    return return_dict