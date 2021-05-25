import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance
import math
from torchvision.ops import boxes as box_ops

def calculate_iou(box1, box2):
    """Calculates IoU between two boxes

    :param box1: first boss in [x,y,w,h] format
    :type box1: list[float, float, float, float]
    :param box2: second boss in [x,y,w,h] format
    :type box2: list[float, float, float, float]
    :return: iou value
    :rtype: float
    """

    # 1.get the coordinate of inters
    ixmin = max(box1[0], box2[0])
    ixmax = min(box1[2] + box1[0], box2[2] + box2[0])
    iymin = max(box1[1], box2[1])
    iymax = min(box1[3] + box1[1], box2[3] + box2[1])
    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)
    # 2. calculate the area of inters
    inters = iw*ih
    # 3. calculate the area of union
    uni = ((box1[2]) * (box1[3]) +
           (box2[2]) * (box2[3]) -
           inters)
    # 4. calculate the overlaps between pred_box and gt_box
    if uni == 0:
        iou = 0
    else:
        iou = inters / uni

    return iou


def extract_all_anchor_boxes(input_shape, fmap_sizes, anchors):
    """Extract anchor boxes on all feature map positions

    :param input_shape: width and height of model's
        input image
    :type input_shape: tuple(int, int)
    :param fmap_sizes: list of width, height tuples
        of all feature maps
    :type fmap_sizes: list((int,int)...)
    :param anchors: contains anchor size for all
        feature maps
    :type anchors: list(list((int,int)...)...)
    :return: all calculated anchor positions
    :rtype: numpy array
    """

    # create receptive window for whole image
    img_h, img_w = input_shape
    receptive_window = np.array([[0, 0, img_w, img_h]])
    anchor_boxes = get_anchor_boxes(
        input_shape, fmap_sizes, anchors, receptive_window)
    return anchor_boxes


def get_anchor_boxes(input_shape, fmap_sizes, anchors, receptive_windows):
    """Get all anchor box positions for a specific
    receptive window

    :param input_shape: width and height of model's
        input image
    :type input_shape: tuple(int, int)
    :param fmap_sizes: list of width, height tuples
        of all feature maps
    :type fmap_sizes: list((int,int)...)
    :param anchors: contains anchor size for all
        feature maps
    :type anchors: list(list((int,int)...)...)
    :param receptive_windows: list of receptive windows with
        dimensions relevant to the input image size. They specify
        where to extract anchor positions.
    :type receptive_windows: list((int,int)...)
    :return: all calculated anchor positions
    :rtype: numpy array
    """

    img_h = input_shape[0]
    img_w = input_shape[1]

    n_fmaps = len(fmap_sizes)

    total_boxes = np.empty(shape=[0, 4])

    # for each feature map
    for fmap_idx in range(n_fmaps):

        # calculate grid cell size

        fmap = fmap_sizes[fmap_idx]
        fmap_anchors = anchors[fmap_idx]

        fmap_h = fmap[0]
        fmap_w = fmap[1]

        step_h = math.ceil(img_h / fmap_h)
        step_w = math.ceil(img_w / fmap_w)

        n_boxes = fmap_anchors.shape[0]
        boxes = np.zeros((fmap_h, fmap_w, n_boxes, 4))

        # for each receptive window
        for window in receptive_windows:

            # calculate overlapping anchors to keep for speedup
            x_start = max(0, math.ceil(window[0] / step_w) - 1)
            y_start = max(0, math.ceil(window[1] / step_h) - 1)
            x_end = min(
                math.ceil((window[0] + window[2]) / step_w) - 1, fmap_w - 1)
            y_end = min(
                math.ceil((window[1] + window[3]) / step_h) - 1, fmap_h - 1)

            # calculate anchor dimenions [x, y, w, h]

            h_len = y_end + 1 - y_start
            w_len = x_end + 1 - x_start

            cy = np.linspace((y_start + 0.5) * step_h,
                             (y_end + 0.5) * step_h, h_len)
            cx = np.linspace((x_start + 0.5) * step_h,
                             (x_end + 0.5) * step_h, w_len)

            cx_grid, cy_grid = np.meshgrid(cx, cy)

            cx_grid = np.expand_dims(cx_grid, -1)
            cy_grid = np.expand_dims(cy_grid, -1)

            # use full-size array to maintain anchor positions

            boxes[y_start:y_end+1, x_start:x_end+1, :, 0] = np.tile(
                cx_grid, (1, 1, n_boxes)) - fmap_anchors[:, 0] // 2
            boxes[y_start:y_end+1, x_start:x_end+1, :, 1] = np.tile(
                cy_grid, (1, 1, n_boxes)) - fmap_anchors[:, 1] // 2
            boxes[y_start:y_end+1, x_start:x_end +
                  1, :, 2] = fmap_anchors[:, 0]
            boxes[y_start:y_end+1, x_start:x_end +
                  1, :, 3] = fmap_anchors[:, 1]

        # reshape and accumulate all boxes found

        boxes = boxes.reshape(-1, 4)
        total_boxes = np.vstack([total_boxes, boxes])

    return total_boxes
