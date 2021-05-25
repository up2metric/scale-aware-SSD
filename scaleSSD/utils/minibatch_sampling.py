import torch

class BaseSampler:
    """Base sampler class for detection minibatch sampling

    :param minibatch_size: total size of sampled minibatches
    :type minibatch_size: int
    """

    def __init__(self, minibatch_size=128):
        """Constructor method
        """

        self.minibatch_size = minibatch_size

    def filter_tensors(self, tensors, mask):
        """Applies specified mask to each tensor in the list
        
        :param tensors: list of torch tensors to apply the mask to
        :type tensors: list[torch.Tensor...]
        :param mask: mask to apply 
        :type mask: bool
        :return: list of filtered tensors
        :rtype: list[torch.Tensor...]
        """

        filt_tensors = list()
        for tensor in tensors:
            filt_tensors.append(tensor[mask])
        return filt_tensors

    def sample_minibatches(self, cls_gts, cls_pred, reg_gts, reg_pred):
        """Sample minibatches by keeping difficult positives and negatives
        
        :param cls_gts: ground truth classification tensor
        :type cls_gts: torch.Tensor
        :param cls_pred: predicted classification tensor
        :type cls_pred: torch.Tensor
        :param reg_gts: ground truth regression tensor
        :type reg_gts: torch.Tensor
        :param reg_pred: predicted regression tensor
        :type reg_pred: torch.Tensor
        :return: tuple of sampled tensors
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        """

        pos_mask = cls_gts[:, :, 0] > 0
        pos_cls_pred, pos_cls_gts, pos_reg_pred, pos_reg_gts = self.filter_tensors(
            [cls_pred, cls_gts, reg_pred, reg_gts], pos_mask)

        neg_mask = cls_gts[:, :, 0] == 0
        neg_cls_pred, neg_cls_gts = self.filter_tensors(
            [cls_pred, cls_gts], neg_mask)

        num_pos_samples = pos_cls_gts.shape[0]
        num_neg_samples = 3 * num_pos_samples

        neg_sort_values = neg_cls_pred[:, 0]
        neg_indices = torch.argsort(neg_sort_values, dim=0)[:num_neg_samples]

        cls_gts_minibatch = torch.cat(
                [pos_cls_gts, neg_cls_gts[neg_indices]], dim=0).type(torch.FloatTensor)[:self.minibatch_size]
        cls_pred_minibatch = torch.cat(
                [pos_cls_pred, neg_cls_pred[neg_indices]], dim=0).type(torch.FloatTensor)[:self.minibatch_size]

        reg_gts_minibatch = pos_reg_gts.type(torch.FloatTensor)
        reg_pred_minibatch = pos_reg_pred.type(torch.FloatTensor)

        return cls_gts_minibatch, cls_pred_minibatch, reg_gts_minibatch, reg_pred_minibatch


class DynamicNegativeSampler(BaseSampler):
    """Dynamic negative sampler class for detection minibatch sampling.
    Keeps sampling positives and missclassified negatives

    :param minibatch_size: total size of sampled minibatches
    :type minibatch_size: int
    """

    def __init__(self, minibatch_size=10000):
        """Constructor method
        """

        super().__init__(minibatch_size=10000)

    def sample_minibatches(self, cls_gts, cls_pred, reg_gts, reg_pred):
        """Sample minibatches by keeping all positives and missclassified negatives
        
        :param cls_gts: ground truth classification tensor
        :type cls_gts: torch.Tensor
        :param cls_pred: predicted classification tensor
        :type cls_pred: torch.Tensor
        :param reg_gts: ground truth regression tensor
        :type reg_gts: torch.Tensor
        :param reg_pred: predicted regression tensor
        :type reg_pred: torch.Tensor
        :return: tuple of sampled tensors
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        """

        pos_mask = cls_gts[:, :, 0] > 0
        pos_cls_pred, pos_cls_gts, pos_reg_pred, pos_reg_gts = self.filter_tensors(
            [cls_pred, cls_gts, reg_pred, reg_gts], pos_mask)

        neg_mask = cls_gts[:, :, 0] == 0
        neg_cls_pred, neg_cls_gts = self.filter_tensors(
            [cls_pred, cls_gts], neg_mask)

        neg_cls_indices = torch.argmax(neg_cls_pred, dim=1)

        # find missclassified negatives
        neg_mcls_mask = neg_cls_indices > 0
        neg_mcls_gts = neg_cls_gts[neg_mcls_mask]
        neg_mcls_pred = neg_cls_pred[neg_mcls_mask]
        
        cls_gts_minibatch = torch.cat(
            [pos_cls_gts, neg_mcls_gts],  dim=0).type(torch.FloatTensor)[:self.minibatch_size]
        cls_pred_minibatch = torch.cat(
            [pos_cls_pred, neg_mcls_pred], dim=0).type(torch.FloatTensor)[:self.minibatch_size]

        # if nothing found calculate loss on negatives
        
        if cls_gts_minibatch.shape[0] == 0:
            cls_gts_minibatch = neg_cls_gts.type(torch.FloatTensor)
            cls_pred_minibatch = neg_cls_pred.type(torch.FloatTensor)

        # keep all positives for bbox loss
        reg_gts_minibatch = pos_reg_gts.type(torch.FloatTensor)
        reg_pred_minibatch = pos_reg_pred.type(torch.FloatTensor)

        return cls_gts_minibatch, cls_pred_minibatch, reg_gts_minibatch, reg_pred_minibatch

