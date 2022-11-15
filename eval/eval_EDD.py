import math
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import argparse
import os

# Input arguments by user
parser = argparse.ArgumentParser(description='PyTorch BioFors EDD Evaluation')
parser.add_argument('--image-categories', nargs='+', default=['Blot/Gel', 'Microscopy', 'Macroscopy', 'FACS'], help='Choose one/all of the following categories - Blot/Gel, Microscopy, Macroscopy, FACS', required=True)
parser.add_argument('--annotation-dir', help='path to annotation file directory', default='./annotation_files/', required=False)
parser.add_argument('--output-dir', help='path to output/predicted image directory', default='predictions/EDD/', required=False)
parser.add_argument('--image-dir', help='path where input images are stored', default='biofors_images/', required=False)
parser.add_argument('--area_th', help='Choose the area percentage to ignore in predicted images', default=0.1, type=float, required=False)

# evaluate the model
class Evaluate_EDD():
    def __init__(self, pdfs, gt, all_pairs, cls_type, output_dir, image_categories, area_th, image_dir):
        # store the inputs and outputs
        self.pdfs = pdfs
        self.gt = gt
        self.all_pairs = all_pairs
        self.cls_type = cls_type
        self.output_dir = output_dir
        self.image_categories = image_categories
        self.image_dir = image_dir
        self.img_tp = 0
        self.img_fp = 0
        self.img_tn = 0
        self.img_fn = 0
        self.tp = np.array([0], dtype=np.int64)
        self.fp = np.array([0], dtype=np.int64)
        self.tn = np.array([0], dtype=np.int64)
        self.fn = np.array([0], dtype=np.int64)
        self.area_th = area_th

    # evaluate the model
    def evaluate_model(self):
        for doi in tqdm(self.pdfs):
            cat_dict = self.all_pairs[doi]
            for cat,pair_list in cat_dict.items():
                for pair in pair_list:
                    rev_pair = pair.split()[1] + ' ' + pair.split()[0]

                    panel1, panel2 = pair.split()

                    if self.cls_type[doi][panel1] not in self.image_categories:
                        continue

                    img_name1 = os.path.join(self.image_dir+doi+'/'+panel1)
                    img_name2 = os.path.join(self.image_dir+doi+'/'+panel2)
                    img1 = cv2.imread(img_name1, 0)
                    img2 = cv2.imread(img_name2, 0)

                    gt_mask1 = np.zeros_like(img1, dtype=np.int)
                    gt_mask2 = np.zeros_like(img2, dtype=np.int)

                    coord1, coord2 = None, None
                    if doi in self.gt and pair in self.gt[doi]:
                        coord1, coord2 = self.gt[doi][pair]

                    if doi in self.gt and rev_pair in self.gt[doi]:
                        coord2, coord1 = self.gt[doi][rev_pair]

                    if coord1 is not None and coord2 is not None:
                        for c in coord1:
                            gt_mask1[c[1]:c[3],c[0]:c[2]] = 1

                        for c in coord2:
                            gt_mask2[c[1]:c[3],c[0]:c[2]] = 1

                    dir_name = pair.replace(' ', '__')
                    pred_mask1 = cv2.imread(os.path.join(self.output_dir+doi+'/'+dir_name+'/'+panel1), 0)
                    pred_mask2 = cv2.imread(os.path.join(self.output_dir+doi+'/'+dir_name+'/'+panel2), 0)

                    if 255 in pred_mask1:
                        pred_mask1 = (pred_mask1/255).astype(np.int)
                    if 255 in pred_mask2:
                        pred_mask2 = (pred_mask2/255).astype(np.int)

                    unique_val1 = list(np.unique(pred_mask1))
                    unique_val2 = list(np.unique(pred_mask2))

                    assert unique_val1==[0,1] or unique_val1==[1,0] or unique_val1==[0] or unique_val1==[1], 'Mask has values other than 0,1: {}'.format(unique_val1)
                    assert unique_val2==[0,1] or unique_val2==[1,0] or unique_val2==[0] or unique_val2==[1], 'Mask has values other than 0,1: {}'.format(unique_val2)

                    assert pred_mask1.shape==gt_mask1.shape,'Shape Mismatch!!'
                    assert pred_mask2.shape==gt_mask2.shape,'Shape Mismatch!!'

                    # if prediction has 5% white pixels, erase them
                    if np.sum(pred_mask1)<self.area_th*(pred_mask1.shape[0]*pred_mask1.shape[1]):
                        pred_mask1 = 0
                    if np.sum(pred_mask2)<self.area_th*(pred_mask2.shape[0]*pred_mask2.shape[1]):
                        pred_mask2 = 0

                    self.tp += np.sum(pred_mask1*gt_mask1) + np.sum(pred_mask2*gt_mask2)

                    fp1 = pred_mask1 - gt_mask1
                    fp1[fp1!=1] = 0
                    fp2 = pred_mask2 - gt_mask2
                    fp2[fp2!=1] = 0
                    self.fp += np.sum(fp1) + np.sum(fp2)

                    fn1 = gt_mask1 - pred_mask1
                    fn1[fn1!=1] = 0
                    fn2 = gt_mask2 - pred_mask2
                    fn2[fn2!=1] = 0
                    self.fn += np.sum(fn1) + np.sum(fn2)

                    tn1 = np.sum((1-pred_mask1)*(1-gt_mask1))
                    tn2 = np.sum((1-pred_mask2)*(1-gt_mask2))
                    self.tn += tn1 + tn2

                    if np.sum(gt_mask1)>0 or np.sum(gt_mask2)>0:
                        if np.sum(pred_mask1)>0 or np.sum(pred_mask2)>0:
                            self.img_tp += 1
                        else:
                            self.img_fn += 1
                    else:
                        if np.sum(pred_mask1)==0 and np.sum(pred_mask2)==0:
                            self.img_tn += 1
                        else:
                            self.img_fp += 1
                                
        return self

    # make a class prediction for one row of data
    def compute_metric(self):
        print('\033[1m' +'Category - ' + '\033[0m', self.image_categories)
        print('\033[1m' +'Image Level Metrics:'+ '\033[0m')
        print('Image - TP: {} \nImage FP: {} \nImage TN: {} \nImage FN: {}'.format(self.img_tp, self.img_fp, self.img_tn, self.img_fn))
        img_precision = self.img_tp/(self.img_tp+self.img_fp)
        img_recall = self.img_tp/(self.img_tp+self.img_fn)
        img_f1 = 2*(img_precision*img_recall)/(img_precision+img_recall)
        img_acc = (self.img_tp+self.img_tn)/(self.img_tp+self.img_tn+self.img_fp+self.img_fn)
        den = 0.5*(np.log(self.img_tp+self.img_fp) + np.log(self.img_tp+self.img_fn) + np.log(self.img_tn+self.img_fp) + np.log(self.img_tn+self.img_fn))
        num = np.log(self.img_tp*self.img_tn - self.img_fp*self.img_fn)
        img_mcc = np.exp(num - den)

        print('Image Acc: {:.4f} \nImage Precision: {:.4f} \nImage Recall: {:.4f} \nImage F1: {:.4f} \nImage MCC: {:.4f}'.format(img_acc, img_precision, img_recall, img_f1, img_mcc))
        print('------------------------------------------------------------------------------------')
        print('\033[1m' +'Pixel Level Metrics:'+ '\033[0m')
        self.tp = self.tp[0]
        self.fp = self.fp[0]
        self.tn = self.tn[0]
        self.fn = self.fn[0]
        print('Pixel TP: {} \nPixel FP: {} \nPixel TN: {} \nPixel FN: {}'.format(self.tp, self.fp, self.tn, self.fn))

        precision = self.tp/(self.tp+self.fp)
        recall = self.tp/(self.tp+self.fn)
        f1 = 2*(precision*recall)/(precision + recall)
        den = 0.5*(np.log(self.tp+self.fp) + np.log(self.tp+self.fn) + np.log(self.tn+self.fp) + np.log(self.tn+self.fn))
        num = np.log(self.tp*self.tn - self.fp*self.fn)
        mcc = np.exp(num - den)
        ts = self.tp/(self.tp+self.fp+self.fn)

        print('Pixel Precision: {:.4f} \nPixel Recall: {:.4f} \nPixel F1: {:.4f} \nPixel MCC: {:.4f} \nPixel TS: {:.4f}'.format(precision, recall, f1, mcc, ts))


    
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # prepare the data
    with open(os.path.join(args.annotation_dir, 'edd_pdfs.pkl'), 'rb') as f:
        pdfs = pickle.load(f)

    with open(os.path.join(args.annotation_dir, 'edd_gt.json'), 'r') as f:
        gt = json.load(f)

    with open(os.path.join(args.annotation_dir, 'edd_test_pairs.json'), 'r') as f:
        all_pairs = json.load(f)

    with open(os.path.join(args.annotation_dir, 'classification.json'), 'r') as f:
        cls_type = json.load(f)
        
    # Initialize the Evaluation class
    init_EDD = Evaluate_EDD(pdfs, gt, all_pairs, cls_type, args.output_dir, args.image_categories, args.area_th, args.image_dir)
    
    # Evaluate the EDD model
    eval_EDD = Evaluate_EDD.evaluate_model(init_EDD)
    
    # Display the metric computation                   
    Evaluate_EDD.compute_metric(eval_EDD)
