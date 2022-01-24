import sys
import numpy as np
import pandas as pd
import os
from collections import Counter
from openpyxl import Workbook

# py mAP_metrics.py ./Video_Dataset/Test_data/labelled_test_data/labelled_test_data/labels/ ./inference_output\test_data\od_frcnn_inference_combine(epoch_72)/labels/

#Specify the file path for ground truth and model predictions respectively
ground_truth_path_folder = sys.argv[1]
predicted_path_folder = sys.argv[2]

def gt_pred_bbox_list(path_to_gt_folder, path_to_pred_folder):
    gt_list = []
    pred_list = []
    class_list = []
    classes = []
    for files in os.listdir(path_to_gt_folder):
        if files.endswith('.txt'):
            gt_file = os.path.join(path_to_gt_folder, files)
            with open(gt_file, 'r') as gt_f:
                for lines in gt_f:
                    gt_annotations = lines.split()
                    #[train_idx, class_pred, xmin, ymin, xmax, ymax]
                    gt_annotation = [files, gt_annotations[0], float(gt_annotations[4]), float(gt_annotations[5]), float(gt_annotations[6]), float(gt_annotations[7])]
                    classes.append(gt_annotations[0])
                    gt_list.append(gt_annotation)
            gt_f.close()
    for files in os.listdir(path_to_pred_folder):
        if files.endswith('.txt'):
            pred_file = os.path.join(path_to_pred_folder, files)
            if os.stat(pred_file).st_size > 0:
                with open(pred_file, 'r') as pred_f:
                    for lines in pred_f:
                        pred_annotations = lines.split()
                        #[train_idx, class_pred, confidence_level, xmin, ymin, xmax, ymax]
                        pred_annotation = [files, pred_annotations[0], float(pred_annotations[15]), float(pred_annotations[4]), float(pred_annotations[5]), float(pred_annotations[6]), float(pred_annotations[7])]
                        pred_list.append(pred_annotation)
                pred_f.close()
    for x in classes:
        if x.lower() not in class_list:
            class_list.append(x.lower())
    
    return gt_list, pred_list, class_list

def intersection_over_union(gt_box, pred_box):
    
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box
    #Raise error when annotations is incorrect
    if (gt_xmin > gt_xmax) or (gt_ymin > gt_ymax):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (pred_xmin > pred_xmax) or (pred_ymin > pred_ymax):
        raise AssertionError("Predicted Bounding Box is not correct")
    
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(gt_xmax < pred_xmin):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        return 0.0, 0.0, 0.0
    
    elif(gt_ymax < pred_ymin): 
        # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        return 0.0, 0.0, 0.0
    
    elif(gt_xmin > pred_xmax): 
        # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        return 0.0, 0.0, 0.0
    
    elif(gt_ymin > pred_ymax): 
        # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0, 0.0, 0.0
    
    else:
        #Calculating the IOU score of pred bbox
        gt_width = gt_xmax - gt_xmin
        gt_height = gt_ymax - gt_ymin
        pred_width = pred_xmax - pred_xmin
        pred_height = pred_ymax - pred_ymin
        gt_bbox = [gt_xmin, gt_ymin, gt_width, gt_height]
        pred_bbox = [pred_xmin, pred_ymin, pred_width, pred_height]

        inter_box_top_left = [max(gt_bbox[0], pred_bbox[0]), max(gt_bbox[1], pred_bbox[1])]
        inter_box_bottom_right = [min(gt_bbox[0] + gt_bbox[2], pred_bbox[0] + pred_bbox[2]), min(gt_bbox[1] + gt_bbox[3], pred_bbox[1] + pred_bbox[3])]

        inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
        inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

        intersection = inter_box_w * inter_box_h
        union = (gt_bbox[2] * gt_bbox[3] + pred_bbox[2] * pred_bbox[3]) - intersection

        iou = intersection / union

        return iou, intersection, union
    
def mean_average_precision(
    path_to_gt_folder, path_to_pred_folder, iou_threshold = 0.5
):
    gt_list, pred_list, class_list = gt_pred_bbox_list(path_to_gt_folder, path_to_pred_folder)
    #format of the pred_boxes (list) : [[train_idx, class_pred, confidence_level, xmin, ymin, xmax, ymax], ...]
    #format of the pred_boxes (list) : [[train_idx, class_pred, xmin, ymin, xmax, ymax], ...]
    average_precision = []
    detection_dictionary = {k:[] for k in ['Filename', 'Class', 'Confidence Level', 'True Positive', 'False Positive', 'Acc TP', 'Acc FP', 'Precision', 'Recall']}
    
    for c in class_list:
        detections = []
        ground_truths = []

        for detection in pred_list:
            if detection[1] == c:
                detections.append(detection)

        for true_box in gt_list:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        #img 0  has 3 bboxes
        #img 1 has 5 bboxes
        #amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = np.zeros(val)
        #amount_boxes = {0:torch.tensor([0,0,0]), 1:torch:tensor([0,0,0,0,0])}(keeping track of the targeted bboxes that we have taken so far)
        
        #Sort the predicted bboxes based on confidence level
        detections.sort(key=lambda x: x[2], reverse = True)

        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        #Go through all detection for the particular class
        for detection_idx, detection in enumerate(detections):
            #Appendinging the respective information into the detection dictionary
            detection_dictionary["Filename"].append(detection[0])
            detection_dictionary["Class"].append(detection[1])
            detection_dictionary["Confidence Level"].append(detection[2])
            #taking out the ground truths that have the same img id as detection img id
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            #number of target bboxes in the image 
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate (ground_truth_img):
                #taking the bbox coordinates to calculate the iou score
                iou, intersection, union = intersection_over_union(detection[3:] ,gt[2:])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                #If the ground truth index of the particular image index have not been covered, then Tp + 1
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    #This bbox index have been covered and we dont want to cover it again in the future
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    #if the GT bbox have been covered previously
                    FP[detection_idx] = 1
            else:
                #Iou score is less then threshold
                FP[detection_idx] = 1
        #Sum up the accumulative values for True Positive False Positive Respectively
        TP_cumsum = np.cumsum(TP,dtype = int)
        FP_cumsum = np.cumsum(FP, dtype = int)

        #Calculating the  accumulative recalls and precision 
        recalls = TP_cumsum/(total_true_bboxes)
        precisions = np.divide(TP_cumsum, (TP_cumsum + FP_cumsum))

        precisions = precisions.tolist()
        recalls = recalls.tolist()
        #Appending the values into the detection dictionary
        for tp in TP: detection_dictionary["True Positive"].append(tp)
        for fp in FP: detection_dictionary["False Positive"].append(fp)
        for acc_tp in TP_cumsum: detection_dictionary["Acc TP"].append(acc_tp)
        for acc_fp in FP_cumsum: detection_dictionary['Acc FP'].append(acc_fp)
        for precision in precisions: detection_dictionary["Precision"].append(precision)
        for recall in recalls: detection_dictionary["Recall"].append(recall)
        #calculating the average precison score for a class
        ap, mrec, mpre, i_list = voc_ap(recalls, precisions)
        average_precision.append(ap)
    #converting the dictionary into a dataframe
    df = pd.DataFrame(data=detection_dictionary)
    label_folder = os.path.dirname(path_to_pred_folder)
    experiment_folder = os.path.dirname(label_folder)
    output = df.to_excel(os.path.join(experiment_folder, 'eval_output.xlsx'))
    return sum(average_precision) / len(average_precision)

def voc_ap(recall, precision):
    
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    recall.insert(0, 0.0) # insert 0.0 at begining of list
    recall.append(1.0) # insert 1.0 at end of list
    mrec = recall[:]
    precision.insert(0, 0.0) # insert 0.0 at begining of list
    precision.append(0.0) # insert 0.0 at end of list
    mpre = precision[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre, i_list

ap = mean_average_precision(ground_truth_path_folder, predicted_path_folder)
print(ap)