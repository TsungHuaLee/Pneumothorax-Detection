# +
import torch.nn as nn
import torch 

class CrossEntropy(nn.Module):
    def __init__(self, weight = None):
        super().__init__()
        self.weight = weight
    @property
    def __name__(self):
        return 'cross entropy'
    def forward(self, y_pred, y_gt):
        if self.weight is None:
            return nn.CrossEntropyLoss()(y_pred, y_gt)
        else:
            self.weight = torch.tensor(self.weight).cuda()
            return nn.CrossEntropyLoss(weight=self.weight)(y_pred, y_gt)

class Fscore(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
        self.threshold = torch.tensor([0.5]).to('cuda')
    @property
    def __name__(self):
        return 'fscore'
    def forward(self, pred, gt):
        pred = torch.softmax(pred, dim = 1)
        pred = torch.argmax(pred, dim = 1)
        
        tp = (gt * pred).sum().to(torch.float32)
        tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
        fp = ((1 - gt) * pred).sum().to(torch.float32)
        fn = (gt * (1 - pred)).sum().to(torch.float32)
        
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        return (2 * precision * recall + self.eps) / (precision + recall + self.eps)
    
class Specificity(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
    @property
    def __name__(self):
        return 'specificity'
    def forward(self, pred, gt):
        pred = torch.softmax(pred, dim = 1)
        pred = torch.argmax(pred, dim = 1)
        
        tp = (gt * pred).sum().to(torch.float32)
        tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
        fp = ((1 - gt) * pred).sum().to(torch.float32)
        fn = (gt * (1 - pred)).sum().to(torch.float32)

        return (tn + self.eps)/(fp + tn + self.eps)
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'accuracy'
    def forward(self, y_pred, y_gt):
        y_pred = torch.softmax(y_pred, dim = 1)
        y_pred = torch.argmax(y_pred, axis = 1)
        tp = torch.sum(y_gt == y_pred, dtype=y_pred.dtype)
        tp = tp.type(torch.DoubleTensor)
        score = tp/y_gt.view(-1).shape[0]
        return score
