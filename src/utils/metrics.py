import torch

def bbox_iou(pred_boxes, true_boxes, eps=1e-7):
    """
    Compute IoU for bboxes in the xywh format
    """
    
    # coordinates of the intersection rectangle
    x1 = torch.max(pred_boxes[..., 0], true_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], true_boxes[..., 1])
    
    x2 = torch.min(pred_boxes[..., 0] + pred_boxes[..., 2], true_boxes[..., 0] + true_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 1] + pred_boxes[..., 3], true_boxes[..., 1] + true_boxes[..., 3])
    
    # intersection area
    intersection_width = torch.clamp(x2 - x1, min=0)
    intersection_height = torch.clamp(y2 - y1, min=0)
    intersection = intersection_width * intersection_height
    
    # union area
    pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
    true_area = true_boxes[..., 2] * true_boxes[..., 3]
    union = pred_area + true_area - intersection
    
    iou = intersection / (union + eps) # add epsilon to avoid division by zero
    
    return iou

def iou_loss(pred_boxes, true_boxes, eps=1e-7):
    """
    Compute the IoU loss
    """
    iou = bbox_iou(pred_boxes, true_boxes, eps)
    return 1 - iou.mean()

    