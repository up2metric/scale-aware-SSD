import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance
import math
from torchvision.ops import boxes as box_ops
from scaleSSD.utils.box_utils import calculate_iou, get_anchor_boxes

class BboxEncoder:
    """This class is responsible for encoding the annotation boxes
    into the format expected by the Custom SSD model

    :param all_anchor_boxes: array containing all the anchor boxes coordinates
    :type all_anchor_boxes: numpy array
    :param input_shape: the image shape expected by the model
    :type input_shape: tuple[int, int]
    :param fmap_sizes: a list of tuples containing the sizes of feature map
        outputs of the model
    :type fmap_sizes: list[tuple[int,int]...]
    :param anchors: list containing the anchors per feature map output
    :type anchors: list[numpy.array...]
    :param no_classes: the number of classes outputed by the model
    :type no_classes: int
    """

    def __init__(self, all_anchor_boxes, input_shape, fmap_sizes, anchors, no_classes):
        """Constructor method
        """

        self.input_shape = input_shape
        self.fmap_sizes = fmap_sizes
        self.anchors = anchors
        self.all_anchor_boxes = all_anchor_boxes
        self.no_classes = no_classes

    def calc_offsets(self, annos, anchors):
        """Calculates dimension offsets of annotations based on anchors

        :param annos: the annotations to offset
        :type annos: numpy.array
        :param anchors: the anchors to use
        :type anchors: numpy.array
        :return: array containing offseted annotations
        :rtype: numpy.array
        """

        offset_annos = np.zeros(annos.shape)
        offset_annos[:, :2] = (annos[:, :2] - anchors[:, :2]) / anchors[:, 2:4]
        offset_annos[:, 2:4] = annos[:, 2:4] / anchors[:, 2:4]
        return offset_annos

    def encode_annos(self, annos, classes, min_thres=0.2, max_thres=0.4):
        """Encodes annotations on fixed-dimension array expected by the model

        :param annos: the annotations to encode
        :type annos: numpy.array
        :param classes: the annotation classes to encode
        :type classes: numpy.array
        :param min_thres: the minimum accepted iou_threshold for assignment,
            defaults to 0.4
        :type min_thres: float, optional
        :param max_thres: the maximum accepted iou_threshold for assignment,
            defaults to 0.7
        :type max_thres: float, optional
        :return: tuple of encoded classification and bounding box offset tensors
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        anchor_boxes = get_anchor_boxes(
            self.input_shape, self.fmap_sizes, self.anchors, annos)

        # create the ground truth arrays
        encoded_cls = np.zeros((anchor_boxes.shape[0], 1))
        encoded_reg = np.zeros(anchor_boxes.shape)

        # find positive (non-zero) anchors (those who are kept based on receptive windows)
        enc_ids = np.where(np.sum(anchor_boxes, axis=1) > 0)[0]

        # if not found, it means there are no annotations
        if enc_ids.shape[0] == 0:
            return torch.from_numpy(encoded_cls), torch.from_numpy(encoded_reg)

        # get positive anchors
        pos_boxes = anchor_boxes[enc_ids]
        
        # calculate iou between annos and positive anchors
        iou_cdist = scipy.spatial.distance.cdist(
            annos, pos_boxes, calculate_iou)
        iou_cdist[iou_cdist < min_thres] = 0

        # keep matches with iou > max_thres

        thres_mask = np.where(iou_cdist >= max_thres,
                              iou_cdist, np.zeros_like(iou_cdist))
        annos_indices, anchor_indices = np.nonzero(thres_mask)

        # keep matches with maximum iou for each annotation
        max_term = np.absolute(
            (iou_cdist - np.max(iou_cdist, axis=1).reshape([iou_cdist.shape[0], 1]))) < 1e-5
        max_mask = np.where(max_term, iou_cdist, np.zeros_like(iou_cdist))
        max_annos_indices, max_anchor_indices = np.nonzero(max_mask)

        # keep maximum iou matches for an anno only if iou > max_thres match
        # wasn't found for that anno

        filtered_max_indices = [(x, y) for x, y in zip(
            max_annos_indices, max_anchor_indices) if x not in annos_indices]

        # if there are maximum iou matches add them to matches
        if len(filtered_max_indices) > 0:
            max_annos_indices, max_anchor_indices = map(
                np.array, zip(*filtered_max_indices))
            annos_indices = np.concatenate([annos_indices, max_annos_indices])
            anchor_indices = np.concatenate(
                [anchor_indices, max_anchor_indices])

        # retrieve matched annos and anchors
        chosen_annos = annos[annos_indices]
        chosen_anchors = pos_boxes[anchor_indices]
        chosen_classes = classes[annos_indices].astype('int')

        # calculate offsets for annos
        offset_annos = self.calc_offsets(chosen_annos, chosen_anchors)

        # insert them into the gt arrays
        encoded_reg[enc_ids[anchor_indices], :] = offset_annos
        encoded_cls[enc_ids[anchor_indices], :] = np.expand_dims(
            chosen_classes, axis=1)

        return torch.from_numpy(encoded_cls), torch.from_numpy(encoded_reg)


class BboxDecoder(nn.Module):
    """This class is responsible for decoding the output of the Custom SSD
    model into bounding boxes. This class is a nn.Module and can be appended
    to the Custom SSD model.

    :param all_anchor_boxes: array containing all the anchor boxes coordinates
    :type all_anchor_boxes: numpy array
    :param input_shape: the image shape expected by the model
    :type input_shape: tuple[int, int]
    :param conf_thres: the minimum confidence threshold that decided
        whether to keep or drop the box, defaults to 0.3
    :type conf_thres: float, optional
    :param max_num: maximum number of boxes to keep,
        defaults to 300
    :type max_num: int, optional
    """

    def __init__(self, all_anchor_boxes, input_shape, conf_thres=0.3, max_num=300):
        """Constructor method
        """

        super().__init__()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.anchor_boxes = torch.tensor(all_anchor_boxes, device=self.device)

        self.input_shape = input_shape
        self.conf_thres = conf_thres
        self.max_num = max_num

    def decalc_offsets(self, offset_annos, anchors):
        """De-calculates dimension offsets of annotations, based on their
        reference anchors

        :param offset_annos: the offseted annotations
        :type offset_annos: torch.Tensor
        :param anchors: the anchors to use
        :type anchors: torch.Tensor
        :return: array containing de-offseted annotations
        :rtype: torch.Tensor
        """

        annos = torch.zeros(offset_annos.shape, device=self.device)
        annos = annos.type(torch.cuda.DoubleTensor)

        annos[:, :2] = offset_annos[:, :2] * anchors[:, 2:4] + anchors[:, :2]
        annos[:, 2:4] = offset_annos[:, 2:4] * anchors[:, 2:4]
        return annos

    def forward(self, x):
        """Forward function for the BboxDecoder layer

        :param x: tuple of torch Tensors, containing the output
            data of the Custom SSD model
        :type x: tuple(torch.Tensor, torch.Tensor)
        :return: tuple containing Custom SSD output data as well as
            the decoded boxes, scores and classes
        :rtype: tuple(torch.Tensor, torch.Tensor, list, list, list)
        """

        final_boxes = list()
        final_scores = list()
        final_classes = list()

        encoded_cls, encoded_reg = x

        # apply softmax to get probabilities
        encoded_cls = torch.nn.functional.softmax(encoded_cls, dim=2)

        # keep boxes and scores only if class found
        batch_scores, classes = torch.max(encoded_cls, dim=2)
        batch_mask = (classes > 0) & (batch_scores > self.conf_thres)
        batch_box_offsets = encoded_reg
        batch_classes = classes

        for scores, box_offsets, classes, mask in zip(batch_scores,
                                                      batch_box_offsets,
                                                      batch_classes,
                                                      batch_mask):

            chosen_scores = scores[mask]
            chosen_box_offsets = box_offsets[mask]
            chosen_classes = classes[mask]
            chosen_anchors = self.anchor_boxes[mask]

            # de-calculate annos offsets to get absolute values
            chosen_boxes = self.decalc_offsets(
                chosen_box_offsets, chosen_anchors)

            final_boxes.append(chosen_boxes[:self.max_num])
            final_scores.append(chosen_scores[:self.max_num])
            final_classes.append(chosen_classes[:self.max_num])

        return encoded_cls, encoded_reg, final_boxes, final_scores, final_classes


class BboxNMS(nn.Module):
    """This class is responsible for applying NMS to the outputs of the
    BboxDecoder layer. This class is a nn.Module and can be appended
    to the Custom SSD model, after BboxDecoder layer.

    :param nms_thres: nms threshold to use,
        defaults to 0.3
    :type nms_thres: float, optional
    """

    def __init__(self, nms_thres=0.3):
        """Constructor method
        """

        super().__init__()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.nms_thres = nms_thres

    def forward(self, x):
        """Forward function for the BboxNMS layer

        :param x: tuple containing Custom SSD output data as well as
            the decoded boxes, scores and classes
        :type x: tuple(torch.Tensor, torch.Tensor, list, list, list)
        :return: tuple containing Custom SSD output data as well as
            the nms filtered boxes, scores and classes
        :rtype: tuple(torch.Tensor, torch.Tensor, list, list, list)
        """

        total_nms_boxes = list()
        total_nms_scores = list()
        total_nms_classes = list()

        batch_encoded_cls, batch_encoded_reg, batch_boxes, batch_scores, batch_classes = x

        for scores, boxes, classes in zip(batch_scores,
                                          batch_boxes,
                                          batch_classes):
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
        
            chosen_ids = box_ops.batched_nms(
                boxes.float(), scores, classes, self.nms_thres)

            nms_scores = scores[chosen_ids]
            nms_boxes = boxes[chosen_ids]
            nms_boxes[:, 2] -= nms_boxes[:, 0]
            nms_boxes[:, 3] -= nms_boxes[:, 1]
            nms_classes = classes[chosen_ids]

            total_nms_boxes.append(nms_boxes)
            total_nms_scores.append(nms_scores)
            total_nms_classes.append(nms_classes)

        return batch_encoded_cls, batch_encoded_reg, total_nms_boxes, total_nms_scores, total_nms_classes
