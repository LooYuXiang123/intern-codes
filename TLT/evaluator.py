import os
import numpy as np 

class evaluator:

    def calculate_AP(self, path_to_gt_folder, path_to_pred_folder):

        print('Starting to calulate Average Precision...')
        ap, mpre, mrec, _ = self.ElevenPointInterpolatedAP(path_to_gt_folder, path_to_pred_folder)

        return ap, mpre, mrec, _

    # Sorting the detections made based on confidence level in descending order
    def sort_cf(self, pred_folder):
        cf_list = []
        detection_count = 0
        for files in os.listdir(pred_folder):
            if files.endswith('txt'):
                pred_file = os.path.join(pred_folder, files)
                with open(pred_file, 'r') as f:
                    for lines in f:
                        detection_count += 1
                        annotations = lines.split()
                        confidence_level = float(annotations[15])
                        detection = {'Detection Number': detection_count, 'File': os.path.basename(pred_file), 'Confidence Level': confidence_level}
                        cf_list.append(detection)
                f.close()
        sorted_cf_list = sorted(cf_list, key = lambda d: d['Confidence Level'], reverse = True)
        
        return sorted_cf_list

    #Calculating the IOU score of each detection (How closely the predicted bbox fit the GT bbox)
    def intersect_over_union(self, gt_box, pred_box):
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
    #Calculating the Accumulative True Positive, False Positive, Precision and Recall
    def calculate_precision_recall(self, gt_folder, pred_folder):
        #Note: Total lenght of gt_list will give tp + fn for recall
        gt_list = []
        for files in os.listdir(gt_folder):
            if files.endswith('.txt'):
                gt_file = os.path.join(gt_folder, files)
                with open(gt_file, 'r') as f:
                    for lines in f:
                        annotations = lines.split()
                        gt_cord = [float(annotations[4]), float(annotations[5]), float(annotations[6]), float(annotations[7])]
                        gt_list.append(gt_cord)
                f.close()
        total_tp = 0
        total_fp = 0
        detection_count = 0
        iou_threshold = 0.5
        sorted_cf_list = self.sort_cf(pred_folder)
        # Calulating Total True positive/False Postive
        for detections in sorted_cf_list:
            tp_count = 0
            fp_count = 0
            filename = detections.get('File')
            gt_label = os.path.join(gt_folder, filename)
            if os.path.exists(gt_label) == False:
                raise AssertionError('Ground Truth Annotation File does not exists')
            pred_label = os.path.join(pred_folder, filename)
            with open(gt_label, 'r') as gt_f:
                for lines in gt_f:
                    gt_annotations = lines.split()
                    gt_box = [float(gt_annotations[4]), float(gt_annotations[5]), float(gt_annotations[6]), float(gt_annotations[7])]
            gt_f.close()
            with open(pred_label, 'r') as pred_f:
                for lines in pred_f:
                    pred_annotations = lines.split()
                    confidence_score = float(pred_annotations[15])
                    if confidence_score == detections.get('Confidence Level'):
                        pred_box = [float(pred_annotations[4]), float(pred_annotations[5]), float(pred_annotations[6]), float(pred_annotations[7])]
                        iou, intersection, union = self.intersect_over_union(gt_box, pred_box)
                        if iou >= iou_threshold:
                            tp_count += 1
                            total_tp += 1
                        else:
                            fp_count += 1
                            total_fp += 1
                        precision = total_tp / (total_tp + total_fp)
                        recall = total_tp / (len(gt_list))
                        calculations = {'True Positive': tp_count, 'False Positive': fp_count, 'Acc TP': total_tp, 'Acc FP': total_fp, 'Precision': precision, 'Recall': recall}
                        detections.update(calculations)
            pred_f.close()
        precision = []
        recall = []
        for detection in sorted_cf_list:
            detection_precision = detection.get('Precision')
            detection_recall = detection.get('Recall')
            precision.append(detection_precision)
            recall.append(detection_recall)
        return precision, recall

    #
    def ElevenPointInterpolatedAP(self, path_to_gt_folder, path_to_pred_folder):
        precision, recall = self.calculate_precision_recall(path_to_gt_folder, path_to_pred_folder)
        mrec = []
        [mrec.append(e) for e in recall]
        mpre = []
        [mpre.append(e) for e in precision]
        recallvalues = np.linspace(0,1,11)
        recallvalues = list(recallvalues[::-1])
        rhointerp = []
        recallvalid =[]
        for r in recallvalues:
            #obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            #if there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallvalid.append(r)
            rhointerp.append(pmax)
        #BY definition Ap=sum(max(precision whose recall is above r))/11
        ap = sum(rhointerp) / 11
        rvals = []
        rvals.append(recallvalid[0])
        [rvals.append(e) for e in recallvalid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhointerp]
        pvals.append(0)
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i-1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
            recallvalues =  [i[0] for i in cc]
            rhointerp = [i[1] for i in cc]
            return [ap, rhointerp, recallvalues, None]
