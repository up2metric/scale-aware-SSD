import torch
import torch.nn as nn
import torch.nn.functional as F
from scaleSSD.utils.minibatch_sampling import *


class ODLoss(nn.Module):
    """Object detection loss class. Has a cross entropy and
        a regression component.

    :param bbox_lamda: regression lamda factor
    :type bbox_lamda: float
    :param minibatch_size: maximum size of the minibatch
    :type minibatch_size: int
    :param cls_lamda: classification lamda factor
    :type cls_lamda: float
    :param sampler: minibatch sampler type. Can be
        'dns' or 'basic'
    :type sampler: str
    :param class_weights: list of weights for each category
    :type class_weights: list[float...] or None
    """

    def __init__(self,
                 minibatch_size,
                 sampler,
                 class_weights):
        """Constructor method
        """

        super(ODLoss, self).__init__()

        self.class_weights = class_weights

        if class_weights:
            self.class_weights = torch.tensor(
                [1] + class_weights).type(torch.FloatTensor)

        if sampler == 'dns':
            self.sampler = DynamicNegativeSampler(minibatch_size)
        elif sampler == 'basic':
            self.sampler = BaseSampler(minibatch_size)

    def calc_ce_loss(self, cls_gts_minibatch, cls_pred_minibatch):
        """Calculates classification cross-entropy loss

        :param cls_gts_minibatch: ground truth minibatch
        :type cls_gts_minibatch: torch.Tensor of shape [n_samples, 1]
        :param cls_pred_minibatch: predicted  minibatch
        :type cls_pred_minibatch: torch.Tensor of shape [n_samples, n_classes]
        :return: cross entropy loss
        :rtype: torch.Tensor
        """

        cls_loss = F.cross_entropy(
            cls_pred_minibatch, cls_gts_minibatch, weight=self.class_weights)
        return cls_loss

    def calc_bbox_loss(self, reg_gts_minibatch, reg_pred_minibatch):
        """Calculates bounding box regression loss

        :param reg_gts_minibatch: ground truth minibatch
        :type reg_gts_minibatch: torch.Tensor of shape [n_samples, 1]
        :param reg_pred_minibatch: predicted  minibatch
        :type reg_pred_minibatch: torch.Tensor of shape [n_samples, n_classes]
        :return: smooth_l1 loss
        :rtype: torch.Tensor
        """

        reg_loss = F.smooth_l1_loss(
            reg_pred_minibatch, reg_gts_minibatch)

        return reg_loss

    # calculate loss
    def forward(self, preds, targets):
        """Forwards loss calculation and returns loss

        :param preds: model predicted predictions
        :type preds: torch.Tensor
        :param targets: annotation targets
        :type targets: torch.Tensor
        :return: total loss (cross entropy + regression loss)
        :rtype: torch.Tensor
        """

        cls_pred, reg_pred = preds[0], preds[1]
        cls_gts,  reg_gts = targets

        cls_pred = torch.nn.functional.softmax(cls_pred, dim=-1)

        cls_gts_minibatch, cls_pred_minibatch, reg_gts_minibatch, reg_pred_minibatch = \
            self.sampler.sample_minibatches(cls_gts, cls_pred, reg_gts, reg_pred)

        cls_gts_minibatch = torch.squeeze(
            cls_gts_minibatch, dim=1).type(torch.LongTensor)

        loss = self.calc_ce_loss(cls_gts_minibatch, cls_pred_minibatch)

        if reg_pred_minibatch.shape[0] > 0:
            reg_loss = self.calc_bbox_loss(
                reg_gts_minibatch, reg_pred_minibatch)
            loss += reg_loss

        return loss
