

import math
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import argparse
import os

# Input arguments by user
parser = argparse.ArgumentParser(description='PyTorch BioFors IDD Evaluation')
parser.add_argument('-pdf_list', metavar='copymove_pdfs', help='path to IDD test pdf list', default='copymove_pdfs')
parser.add_argument('-OUT_DIR', metavar='OUT_DIR', help='path to output/predicted image directory', default='predict/IDD/')
parser.add_argument('-categories', metavar='categories', nargs='+', help='Choose one/all of the following categories - Microscopy, Blot/Gel, Macroscopy', default=['Microscopy', 'Blot/Gel', 'Macroscopy'], required=True)
parser.add_argument('-IDD_gt', metavar='from_scratch_copymove_gt.json', help='path to groundtruth annotation file')
parser.add_argument('-IDD_classification', metavar='from_scratch_classification.json', help='path to image classification annotation')
parser.add_argument('--img_path', metavar='from_scratch_panels/', help='path where input images are stored', default='from_scratch_panels/')

# evaluate the model
class Evaluate_IDD():
    def __init__(self, pdfs, gt, cls_type, OUT_DIR, categories, img_path):
        # store the inputs and outputs
        self.pdfs = pdfs
        self.gt = gt
        self.cls_type = cls_type
        self.OUT_DIR = OUT_DIR
        self.categories = categories
        self.img_path = img_path
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

            for panel, cls in self.cls_type[doi].items():
                if cls in self.categories:
                    
                    img_name = os.path.join(self.img_path+doi+'/'+panel)
                    img = cv2.imread(img_name, 0)

                    gt_mask = np.zeros_like(img, dtype=np.int)

                    coord = None
                    if doi in self.gt and panel in self.gt[doi]:
                        coord = self.gt[doi][panel]

                    if coord is not None:
                        for c in coord:
                            gt_mask[c[1]:c[3],c[0]:c[2]] = 1

                    pred_mask = cv2.imread(os.path.join(self.OUT_DIR+doi+'/'+panel), 0)
                    if pred_mask is None:
                        print(doi, panel)
                        pred_mask = np.zeros_like(img)
                    if 255 in list(np.unique(pred_mask)):
                        pred_mask = pred_mask/255

                    unique_vals = list(np.unique(pred_mask))
                    assert unique_vals==[0,1] or unique_vals==[1,0] or unique_vals==[0] or unique_vals==[1], 'Mask has values other than 0,1: {}'.format(unique_vals)

                    assert pred_mask.shape==gt_mask.shape,'Shape Mismatch!!'
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
        print('\033[1m' +'Category - ' + '\033[0m', self.categories)
        print('\033[1m' +'Image Level Metrics:'+ '\033[0m')
        print('Image TP: {} \nImage FP: {} \nImage TN: {} \nImage FN: {}'.format(self.img_tp, self.img_fp, self.img_tn, self.img_fn))
        img_precision = self.img_tp/(self.img_tp+self.img_fp)
        img_recall = self.img_tp/(self.img_tp+self.img_fn)
        img_f1 = 2*(img_precision*img_recall)/(img_precision+img_recall)
        img_acc = (self.img_tp+self.img_tn)/(self.img_tp+self.img_tn+self.img_fp+self.img_fn)
        den = 0.5*(np.log(self.img_tp+self.img_fp) + np.log(self.img_tp+self.img_fn) + np.log(self.img_tn+self.img_fp) + np.log(self.img_tn+self.img_fn))
        num = np.log(self.img_tp*self.img_tn - self.img_fp*self.img_fn)
        img_mcc = np.exp(num - den)

        print('Image Acc: {:.4f} \nImage Precision: {:.4f} \nImage Recall: {:.4f} \nImage F1: {:.4f} \nImage MCC: {:.4f}'.format(img_acc, img_precision, img_recall, img_f1, img_mcc))
        
        self.tp = self.tp[0]
        self.fp = self.fp[0]
        self.tn = self.tn[0]
        self.fn = self.fn[0]
        print('------------------------------------------------------------------------------------')
        print('\033[1m' +'Pixel Level Metrics:'+ '\033[0m')
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
    pdf_list = args.pdf_list
    OUT_DIR = args.OUT_DIR   
    categories = args.categories
    img_path = args.img_path
    
    with open(pdf_list+'.pkl', 'rb') as f:
        pdfs = pickle.load(f)

    with open(args.IDD_gt, 'r') as f:
        gt = json.load(f)

    with open(args.IDD_classification, 'r') as f:
        cls_type = json.load(f)
    
    # Initialize the Evaluation class
    init_IDD = Evaluate_IDD(pdfs, gt, cls_type, OUT_DIR, categories, img_path)
    
    # Evaluate the IDD model
    eval_IDD = Evaluate_IDD.evaluate_model(init_IDD)
    
    # Display the metric computation                   
    Evaluate_IDD.compute_metric(eval_IDD)
        