import numpy as np
from tqdm import tqdm

def intersection(target, prediction):
    return np.logical_and(target, prediction)

def union(target, prediction):
    return np.logical_or(target, prediction)

def iou_score(target, prediction):
    return np.sum(intersection(target,prediction)) / np.sum(union(target,prediction))

def status(target_mask, pred_mask, iou_threshold):
    score = iou_score(target_mask, pred_mask)
    if (score > iou_threshold) :
        return 'TP'
    elif (score > 0):
        return 'FN'
    else:
        return 'FP'

def status_sum(target_mask_list, pred_mask_list, iou_threshold):
    tp_score = 0
    fn_score = 0
    fp_score = 0
    for target in target_mask_list:
        for pred in pred_mask_list:
            status_val = status(target, pred, iou_threshold)
            if status_val == 'TP':
                tp_score += 1
            elif status_val == 'FN':
                fn_score += 1
            else :
                fp_score += 1
                
    return tp_score, fn_score, fp_score

def precision(tp, fn, fp):
    if ((tp+fp)==0):
        return 0 
    
    return tp/(tp+fp)

def recall(tp, fn, fp):
    if ((tp+fn)==0):
        return 0 
    return tp/(tp+fn)

def max_precision_operation(relation):

    keys = list(relation.keys())
    keys.sort()
    
    for i in range (len(keys)):
        for j in range(i, len(keys)):
            current_val = relation[keys[i]]
            temp = relation[keys[j]]
            if current_val<temp:
                relation[keys[i]] = temp
    
    return relation

def recall_precision_sum(relations):
    
    relation = max_precision_operation(relations)
    keys = list(relation.keys())
    keys.sort()
    score = 0
    
    for i in range(len(keys)-1):
        x1 = keys[i]
        x2 = keys[i+1]
        y1 = relation[x1]
        y2 = relation[x2]
        temp = y2*(x2-x1)-(y2-y1)*(x2-x1)/2
        score += temp
    
    return score

def make_status_dic(target_mask_list, pred_mask_list, threshold):
    result = {}
    tp, fn, fp = status_sum(target_mask_list, pred_mask_list, threshold)
    result['tp'] = tp
    result['fn'] = fn
    result['fp'] = fp
    return [result]
    


def mask_rcnn_metric_one_picture(target, predict, label_list, threshold):
    
    target_fit_list, target_mask = target[0], target[1]
    pred_fit_list, pred_mask = predict[0], predict[1]
    
    score_correct = {}
    
    for i in range (len(label_list)):
        
        target_label = label_list[i]
        target_list = []
        pred_list = []
        
        if (target_label == 'BG'):
            continue 
        
        for j in range (len(target_fit_list)):
            target_fit = target_fit_list[j].item()
            if (target_fit == i):
                target_list += [target_mask[:,:,j]]
        
        for j in range (len(pred_fit_list)):
            pred_fit = pred_fit_list[j].item()
            if (pred_fit == i):
                pred_list += [pred_mask[:,:,j]]
        
        score_correct[target_label] = make_status_dic(target_list, pred_list, threshold) 
    
    return score_correct

def mask_rcnn_metric_one_iou(target_predict_list, label_list, threshold):
    
    score_com = {}
    for label in label_list:
        if label=='BG':
            continue
        score_com[label] = [{'tp':0, 'fn':0, 'fp':0}]
    
    for pair in target_predict_list:
        target = pair[0]
        predict = pair[1]
        
        eval_one = mask_rcnn_metric_one_picture(target, predict, label_list, threshold)
        
        for label in label_list:
            if (label=='BG'):
                continue
            origin_tp = score_com[label][0]['tp']
            origin_fn = score_com[label][0]['fn']
            origin_fp = score_com[label][0]['fp']
            
            temp_tp = eval_one[label][0]['tp']
            temp_fn = eval_one[label][0]['fn']
            temp_fp = eval_one[label][0]['fp']
            
            com_tp = origin_tp + temp_tp
            com_fn = origin_fn + temp_fn
            com_fp = origin_fp + temp_fp
            
            score_com[label] = [{'tp' : com_tp, 'fn' : com_fn, 'fp' : com_fp}]
    
    
    return score_com

def mask_rcnn_metric(target_predict_list, label_list):
    all_score = []
    final_score = {}
    result = {}
    
    for i in tqdm(range (10)):
        threshold = 0.5+i*0.05
        all_score += [mask_rcnn_metric_one_iou(target_predict_list, label_list, threshold)]
    
    for label in label_list:
        if label=='BG':
            continue
        
        recall_prediction_relation = {}
        
        for i in range (10):
            tp,fn, fp = all_score[i][label][0]['tp'], all_score[i][label][0]['fn'], all_score[i][label][0]['fp']
            
            precision_val = precision(tp, fn, fp)
            recall_val = recall(tp, fn, fp)

            # print("{} on {} iou value - TP:{} FN:{} FP:{}".format(label, 0.5+i*0.05, tp, fn, fp))
            recall_prediction_relation[recall_val]=precision_val
        
        # print(recall_prediction_relation)
        result[label] = recall_precision_sum(recall_prediction_relation)
        
    return result


def make_log_com(data_list):
    
    result = data_list[0]
    for data in data_list:
        result = union(result, data)
    
    return result

def sementic_seg_score(target_list, pred_list):
    
    target_com = make_log_com(target_list)
    pred_com = make_log_com(pred_list)
    
    return iou_score(target_com, pred_com)

def sementic_seg_metric(target_predict_list, label_list):
    
    result= {}
    
    for label in label_list:
        if (label == 'BG'):
            continue
        result[label]=0
    
    for pair in tqdm(target_predict_list):
        target = pair[0]
        predict = pair[1]
        
        target_fit_list, target_mask = target[0], target[1]
        pred_fit_list, pred_mask = predict[0], predict[1]
        
        score = {}
        
        for i in range (len(label_list)):
        
            target_label = label_list[i]
            target_list = []
            pred_list = []
            
            a = np.zeros(shape=target_mask[:,:,0].shape)
            b = np.ones(shape=a.shape)
            base = intersection(a,b)
            
            
            if (target_label == 'BG'):
                continue 

            for j in range (len(target_fit_list)):
                target_fit = target_fit_list[j].item()
                if (target_fit == i):
                    target_list += [target_mask[:,:,j]]

            for j in range (len(pred_fit_list)):
                pred_fit = pred_fit_list[j].item()
                if (pred_fit == i):
                    pred_list += [pred_mask[:,:,j]]
            
            if (len(target_list)==0):
                target_list = [base]
                
            if (len(pred_list)==0):
                pred_list = [base]
            
            score[target_label] = sementic_seg_score(target_list, pred_list)
            
        for label in label_list:
            if (label == 'BG'):
                continue
            
            temp = result[label]
            result[label] = temp + score[label]
    
    keys = list(result.keys())
    for key in keys:
        temp = result[key] 
        result[key] = temp/len(target_predict_list)
    
    return result
        
    
        