import torch
import torch.nn as nn
from torch.autograd import Variable

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def compute_loss(output, reports_ids, reports_masks, One_loss=None, Two_loss=None):
    criterion = LanguageModelCriterion()
    lm_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    total_loss = lm_loss
    if One_loss is not None:
        total_loss += One_loss
    if Two_loss is not None:
        total_loss += Two_loss

    return total_loss