import torch

def iou_score(pred, mask, num_classes=50):
    pred = torch.argmax(pred,dim=1)

    ious=[]
    for cls in range(num_classes):
        pred_inds = (pred==cls)
        target_inds = (mask==cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union==0:
            continue
        ious.append((intersection+1e-6)/(union+1e-6))

    return sum(ious)/len(ious)
