import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np
from scaleSSD.utils.box_utils import extract_all_anchor_boxes
from scaleSSD.models.bbox_mapper import BboxEncoder, BboxDecoder, BboxNMS 

class CustomSSD(nn.Module):
    """Describes the custom ssd model architecture

    :param k_list: list containing the number of anchors 
        for each output feature map
    :type k_list: list[int...]
    :param no_classes: the number of classes for the model to output
    :type no_classes: int
    :param out_filters: the number of filters for each output branch
    :type out_filters: list[int...]
    """

    def __init__(self, k_list=[4, 6, 6, 4], no_classes=4, out_filters=[32, 64, 512, 256]):
        """Constructor method
        """

        super().__init__()

        self.fmap_reduce_factors = [8, 16, 32, 64]

        assert len(k_list) == len(out_filters)
        assert len(k_list) == len(self.fmap_reduce_factors)

        self.k_list = k_list
        self.no_classes = no_classes
        self.out_filters = out_filters

        # mobilenet branches
        self.mnet_branch_1 = models.mobilenet_v2(pretrained=True).features[:7]
        self.mnet_branch_2 = models.mobilenet_v2(
            pretrained=True).features[7:11]

        self.conv1_1 = nn.Conv2d(64, 256, 1, padding=0)
        self.conv1_2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(512, 128, 1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        # outs
        for k, f in zip(self.k_list, self.out_filters):
            cls_branch = nn.Conv2d(f, self.no_classes * k, 3, padding=1)
            reg_branch = nn.Conv2d(f, 4 * k, 3, padding=1)
            self.cls_branches.append(cls_branch)
            self.reg_branches.append(reg_branch)

    def forward(self, x):
        """Forward pass of the model. Creates the output based on the input

        :param x: the input tensor to forward to the model
        :type x: torch.Tensor
        :return: a tuple containing the output classification and regression Tensors
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        fmap1 = self.mnet_branch_1(x)

        fmap2 = self.mnet_branch_2(fmap1)

        x = F.relu(self.conv1_1(fmap2))
        fmap3 = F.relu(self.conv1_2(x))

        x = F.relu(self.conv2_1(fmap3))
        fmap4 = F.relu(self.conv2_2(x))

        cls_outs = list()
        reg_outs = list()
        fmaps = [fmap1, fmap2, fmap3, fmap4]

        # --------------- outs -------------------

        for k, cls_branch, reg_branch, fmap in zip(self.k_list, self.cls_branches, self.reg_branches, fmaps):
            cls_branch_out = cls_branch(fmap)
            reg_branch_out = reg_branch(fmap)
            cls_out_inst, reg_out_inst = od_head_reshape(
                cls_branch_out, reg_branch_out, k, self.no_classes)
            cls_outs.append(cls_out_inst)
            reg_outs.append(reg_out_inst)

        cls_out = torch.cat(cls_outs, dim=1)
        reg_out = torch.cat(reg_outs, dim=1)

        return cls_out, reg_out

def od_head_reshape(cls_branch, reg_branch, k, no_classes):
    """Reshapes object detection output feature maps so that they
    can be concatenated
    
    :param cls_branch: the classification output branch to reshape
    :type cls_branch: torch.Tensor
    :param reg_branch: the regression output branch to reshape
    :type reg_branch: torch.Tensor
    :param k: the number of outputed anchors in the feature map
    :type k: int
    :param no_classes: the number of outputed classes
    :type no_classes: int
    :return: a tuple containing the reshaped classification and regression Tensors
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    cls_branch = cls_branch.permute(0, 2, 3, 1)
    reg_branch = reg_branch.permute(0, 2, 3, 1)

    cls_out = cls_branch.reshape([
        cls_branch.shape[0], cls_branch.shape[1] * cls_branch.shape[2] * k, no_classes])
    reg_out = reg_branch.reshape([
        reg_branch.shape[0], reg_branch.shape[1] * reg_branch.shape[2] * k, 4])

    return cls_out, reg_out

def create_detection_model(anchors, no_classes, input_size):
    """Build the detection custom SSD model
    
    :param anchors: list of anchors per fmap
    :type anchors: list
    :param no_classes: the number of outputed classes
    :type no_classes: int
    :param input_size: a tuple describing the input size of an image
    :type input_size: tuple(int, int)
    :return: a tuple containing the model layers and the bbox encoder
    :rtype: tuple(nn.Module, nn.Module, nn.Module), BboxEncoder
    """

    # create anchor arrays
    anchor_list = list()
    for fmap_anchors in anchors:
        fmap_list = list()
        for anchor in fmap_anchors:
            fmap_list.append([anchor, anchor])
        anchor_list.append(np.array(fmap_list))

    # get anchor dim list
    k_list = [x.shape[0] for x in anchor_list]

    model = CustomSSD(k_list, no_classes)

    fmaps = list()
    for reduce_factor in model.fmap_reduce_factors:
        fmap_h = math.ceil(input_size[0] / reduce_factor)
        fmap_w = math.ceil(input_size[1] / reduce_factor)
        fmaps.append([fmap_h, fmap_w])


    all_anchor_boxes = extract_all_anchor_boxes(input_size, fmaps, anchor_list)

    # initialize bbox mapper
    bbox_encoder = BboxEncoder(all_anchor_boxes, input_size, fmaps, anchor_list, no_classes)
    bbox_decoder = BboxDecoder(all_anchor_boxes, input_size)
    bbox_nms = BboxNMS()

    return (model, bbox_decoder, bbox_nms), bbox_encoder
