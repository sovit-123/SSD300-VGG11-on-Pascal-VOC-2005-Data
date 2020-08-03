"""
NOTE:
This utils file is for evaluation purposes if the original dataset deos not
contain object difficulties. The PASCAL VOC 2005 dataset does not contain
object difficulties.
"""

import torch
import json

from utils import label_map

# to save json files
def save_json(file, name):
    with open(f"../json_outputs/{str(name)}.json", 'w') as json_file:
            json.dump(file, json_file)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
    # find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def calc_mAP(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    assert (len(gt_boxes) == len(gt_labels) == len(pred_boxes) == len(pred_labels) == len(pred_scores))
    
    gt_image_ix = []
    pred_image_ix = []
    
    for ix, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        assert(box.size(0) == label.size(0))
        gt_image_ix.extend([ix] * box.size(0))
    for ix, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        assert(box.size(0) == label.size(0) == score.size(0))
        pred_image_ix.extend([ix] * box.size(0))
    
    
    gt_image_ix = torch.tensor(gt_image_ix)
    gt_boxes = torch.cat(gt_boxes, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)
    
    pred_image_ix = torch.tensor(pred_image_ix)
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0)
    
    n_classes = len(label_map)
    AP = torch.zeros((n_classes - 1,), dtype=torch.float)
    
    for class_id in range(1, n_classes):
        # all ground truth box & label associated with this class ( across all the images )
        gt_ix = gt_labels == class_id
        gt_image_ix_class = gt_image_ix[gt_ix]
        gt_boxes_class = gt_boxes[gt_ix]
        gt_labels_class = gt_labels[gt_ix]
        
        # get all predictions for this class and sort them
        pred_ix = pred_labels == class_id
        _, pred_sorted_ix = torch.sort(pred_scores[pred_ix], dim=0, descending=True)
        
        pred_image_ix_class = pred_image_ix[pred_ix][pred_sorted_ix]
        pred_boxes_class = pred_boxes[pred_ix][pred_sorted_ix]
        pred_labels_class = pred_boxes[pred_ix][pred_sorted_ix]
        pred_scores_class = pred_scores[pred_ix][pred_sorted_ix]
        
        # number of detections for this class for this image
        n_class_detections = pred_boxes_class.size(0)
        if n_class_detections == 0: continue
    
        TP = torch.zeros((n_class_detections,))
        FP = torch.zeros((n_class_detections,))
        # to keep track gt_boxes that have already been detected.
        detected_gt_boxes = torch.zeros(gt_image_ix_class.size(0))
        
        for d in range(n_class_detections):
            this_image_ix = pred_image_ix_class[d]
            this_box = pred_boxes_class[d]
            
            # get all gt_boxes in this image which have the same class
            obj_same_class_in_image = gt_boxes_class[gt_image_ix_class == this_image_ix]
            # if no GT box exists for this class for this image, mark FP
            if obj_same_class_in_image.size(0) == 0:
                FP[d] = 1
                continue
            
            # find overlap of this detection with all gt boxes of the same class in this image
            overlap = find_jaccard_overlap(this_box.unsqueeze(0), obj_same_class_in_image)
            max_overlap, ind = torch.max(overlap.squeeze(0), dim=0)
            # index of box in gt_boxes_class with maximum overlap
            gt_matched_index = torch.LongTensor(range(gt_boxes_class.size(0)))[gt_image_ix_class == this_image_ix][ind]
            
            
            if max_overlap.item() > 0.5:
                # if this object has not already been detected, it's a TP
                if detected_gt_boxes[gt_matched_index] == 0:
                    TP[d] = 1
                    detected_gt_boxes[gt_matched_index] = 1 # this gt_box has been detected
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
                
        # compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_TP = torch.cumsum(TP, dim=0)  # (n_class_detections)
        cumul_FP = torch.cumsum(FP, dim=0)  # (n_class_detections)
        cumul_precision = cumul_TP / (cumul_TP + cumul_FP + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_TP / n_class_detections  # (n_class_detections)
    
        # find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        AP[class_id - 1] = precisions.mean()  # c is in [1, n_classes - 1]
    
    recall = cumul_recall.mean().item()
    mAP = AP.mean().item()    
    return AP, mAP, recall