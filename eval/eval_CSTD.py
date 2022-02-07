

import math
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import argparse
import os

# Input arguments by user
parser = argparse.ArgumentParser(description='PyTorch BioFors CSTD Evaluation')
parser.add_argument('-pdf_list', metavar='cstd_pdfs.pkl', help='path to test pdf list', default='cstd_pdfs.pkl')
parser.add_argument('-OUT_DIR', metavar='OUT_DIR', help='path to output/predicted image directory', default='predict/CSTD/')
parser.add_argument('-CSTD_gt', metavar='cstd_gt.json', help='path to groundtruth annotation file', default='cstd_gt.json')
parser.add_argument('-CSTD_classification', metavar='classification.json', help='path to image classification annotation', default='classification.json')
parser.add_argument('--img_path', metavar='from_scratch_panels/', help='path where input images are stored', default='from_scratch_panels/')
parser.add_argument('--label', metavar='label', help='Choose one of the following categories - Microscopy, Blot/Gel, Macroscopy')

# evaluate the model
class Evaluate_CSTD():
    def __init__(self, pdfs, gt, cls_type, OUT_DIR, img_path, label):
        # store the inputs and outputs
        self.pdfs = pdfs
        self.gt = gt
        self.cls_type = cls_type
        self.OUT_DIR = OUT_DIR
        self.img_path = img_path
        self.label = label
        self.img_tp = 0
        self.img_fp = 0
        self.img_tn = 0
        self.img_fn = 0
        self.tp = np.array([0], dtype=np.int64)
        self.fp = np.array([0], dtype=np.int64)
        self.tn = np.array([0], dtype=np.int64)
        self.fn = np.array([0], dtype=np.int64)
        
    def evaluate_model(self):
        for doi in tqdm(self.pdfs):
            img_list = [img for img, label in self.cls_type[doi].items() if label==self.label]
            for panel in img_list:

                img_name = os.path.join(self.img_path+doi+'/'+panel)
                img = cv2.imread(img_name, 0)

                gt_mask = np.zeros_like(img, dtype=np.int)
                if doi in self.gt and panel in self.gt[doi]:
                    coords = self.gt[doi][panel]
                    for c in coords:
                        gt_mask[c[1]:c[3],c[0]:c[2]] = 1

                pred_mask = cv2.imread(os.path.join(self.OUT_DIR+doi+'/'+panel), 0)

                if 255 in list(np.unique(pred_mask)):
                    pred_mask = (pred_mask/255)
                if list(np.unique(pred_mask))==[None]:
                    continue
                assert list(np.unique(pred_mask))==[0,1] or list(np.unique(pred_mask))==[1,0] or list(np.unique(pred_mask))==[0] or list(np.unique(pred_mask))==[1], 'Mask values other than 0,1: {}'.format(np.unique(pred_mask))
                assert pred_mask.shape==gt_mask.shape, '{} {} {} {}'.format(doi, panel, pred_mask.shape, gt_mask.shape)

                pred_mask = pred_mask.astype(np.int64)

                self.tp += np.sum(pred_mask*gt_mask)

                fp1 = pred_mask - gt_mask
                fp1[fp1!=1] = 0
                self.fp += np.sum(fp1)

                fn1 = gt_mask - pred_mask
                fn1[fn1!=1] = 0
                self.fn += np.sum(fn1)

                self.tn += np.sum((1-pred_mask)*(1-gt_mask))

                if np.sum(gt_mask)>0:
                    if np.sum(pred_mask)>0:
                        self.img_tp += 1
                    else:
                        self.img_fn += 1
                else:
                    if np.sum(pred_mask)==0:
                        self.img_tn += 1
                    else:
                        self.img_fp += 1
                        
        return self
                        

    # make a class prediction for one row of data
    def compute_metric(self):
        img_precision = self.img_tp/(self.img_tp+self.img_fp)
        img_recall = self.img_tp/(self.img_tp+self.img_fn)
        img_f1 = 2*(img_precision*img_recall)/(img_precision+img_recall)
        img_acc = (self.img_tp+self.img_tn)/(self.img_tp+self.img_tn+self.img_fp+self.img_fn)
        den = 0.5*(np.log(self.img_tp+self.img_fp) + np.log(self.img_tp+self.img_fn) + np.log(self.img_tn+self.img_fp) + np.log(self.img_tn+self.img_fn))
        num = np.log(self.img_tp*self.img_tn - self.img_fp*self.img_fn)
        img_mcc = np.exp(num - den)

        print('\033[1m' +'Image Level Metrics:'+ '\033[0m')
        print('Image Acc: {:.4f} \nImage Precision: {:.4f} \nImage Recall: {:.4f} \nImage F1: {:.4f} \nImage MCC: {:.4f}'.format(img_acc, img_precision, img_recall, img_f1, img_mcc))
        print('TP: {} \nFP: {} \nTN: {} \nFN: {}'.format(self.tp, self.fp, self.tn, self.fn))

        print('------------------------------------------------------------------------------------')
        print('\033[1m' +'Pixel Level Metrics:'+ '\033[0m')
        precision = (self.tp/(self.tp+self.fp))[0]
        recall = (self.tp/(self.tp+self.fn))[0]
        f1 = 2*(precision*recall)/(precision + recall)
        den = 0.5*(np.log(self.tp+self.fp) + np.log(self.tp+self.fn) + np.log(self.tn+self.fp) + np.log(self.tn+self.fn))
        num = np.log(self.tp*self.tn - self.fp*self.fn)
        mcc = np.exp(num - den)[0]
        print('Pixel Precision: {:.4f} \nPixel Recall: {:.4f} \nPixel F1: {:.4f} \nPixel MCC: {:.4f}'.format(precision, recall, f1, mcc))
           
        
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # prepare the data
    OUT_DIR = args.OUT_DIR
    img_path = args.img_path
    label = args.label
    
    with open(args.pdf_list, 'rb') as f:
        pdfs = pickle.load(f)

    with open(args.CSTD_gt, 'r') as f:
        gt = json.load(f)

    with open(args.CSTD_classification, 'r') as f:
        cls_type = json.load(f)
    
    # Initialize the Evaluation class
    init_CSTD = Evaluate_CSTD(pdfs, gt, cls_type, OUT_DIR, img_path, label)
    
    # Evaluate the CSTD model
    eval_CSTD = Evaluate_CSTD.evaluate_model(init_CSTD)
    print(eval_CSTD.img_tp)
    
    # Display the metric computation                   
    Evaluate_CSTD.compute_metric(eval_CSTD)
    
    
