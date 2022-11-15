import torch
import pickle
import json
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset

class MonetTestDL(Dataset):

    def __init__(self,
                 annotation_dir='./annotation_files/',
                 image_dir='./biofors_images/',
                 image_categories=['Microscopy', 'Blot/Gel', 'Macroscopy', 'FACS']):

        self.image_categories = image_categories
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        with open(os.path.join(self.annotation_dir,'edd_pdfs.pkl'), 'rb') as f:
            self.pdfs = pickle.load(f)

        with open(os.path.join(self.annotation_dir, 'edd_gt.json'), 'r') as f:
            self.gt = json.load(f)

        with open(os.path.join(self.annotation_dir, 'edd_test_pairs.json'), 'r') as f:
            self.all_pairs = json.load(f)

        with open(os.path.join(self.annotation_dir, 'classification.json'), 'r') as f:
            self.cls_type = json.load(f)

        self.sample_list = self.load_samples()
        
    def load_samples(self):

        sample_list = []
        for doi in tqdm(self.pdfs):
            cat_dict = self.all_pairs[doi]
            for cat,pair_list in cat_dict.items():
                for pair in pair_list:
                    rev_pair = pair.split()[1] + ' ' + pair.split()[0]

                    panel1, panel2 = pair.split()

                    if self.cls_type[doi][panel1] not in self.image_categories:
                        continue

                    sample_list += [(doi, panel1, panel2, self.cls_type[doi][panel1])]
                        
        return sample_list

    def preprocess_img(self, img):

        prep_img = cv2.resize(img, (256,256))
        prep_img = prep_img/255.
        prep_img = (prep_img-0.5)/0.5
        prep_img = torch.tensor(prep_img).permute(2,0,1)

        return prep_img

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, idx):

        doi, panel1, panel2, cat = self.sample_list[idx]
        img1 = cv2.imread(os.path.join(self.image_dir, doi, panel1))
        img2 = cv2.imread(os.path.join(self.image_dir, doi, panel2))
        
        s1 = torch.tensor(img1.shape)
        s2 = torch.tensor(img2.shape)
        
        img1 = self.preprocess_img(img1)
        img2 = self.preprocess_img(img2)

        return img1.float(), img2.float(), s1, s2, doi, panel1, panel2
