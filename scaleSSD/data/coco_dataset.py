import os
import cv2
import numpy as np
import random
from scaleSSD.utils.io_ops import json_to_dict
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(Dataset):
    """This is a class that maps a coco formatted dataset
    to the custom ssd model rgb input

    :param step: the train procedure step (train, val, test)
    :type step: str
    :param annos_path: the path to the annotations file
    :type annos_path: str
    :param image_dir: the path to the images directory
    :type image_dir: str
    :param input_size: tuple containing the image size 
        expected by the model
    :type input_size: tuple[int,int]
    :param annos_encoder: instance of BboxEncoder, responsible
        for encoding annotation data
    :type annos_encoder: BboxEncoder
    :param random_seed: the random seed to initialize dataset spliting,
        defaults to 0
    :type random_seed: int, optional
    :param transform_norm_parameters: Parameters for normalization in transforms
    :type transform_norm_parameters: list
    """

    def __init__(self,
                 step,
                 annos_path,
                 image_dir,
                 no_classes,
                 input_size,
                 annos_encoder,
                 random_seed,
                 transform_norm_parameters):
        """Constructor method
        """
        
        self.image_path = image_dir
        self.annos_path = annos_path
        self.step = step
        self.input_size = input_size
        self.annos_encoder = annos_encoder
        self.no_classes = no_classes

        transform_list = list()
        transform_list.append(A.Resize(input_size[0], input_size[1]))
        transform_list.append(A.Normalize(
            *transform_norm_parameters))
        transform_list.append(ToTensorV2(transpose_mask=True))

        self.transformer = A.Compose(
            transform_list,
            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        self.id_to_img_dict, self.id_to_anno_dict = _parse_coco_annos(
            annos_path)
        total_ids = list(self.id_to_img_dict.keys())

        if self.step != 'test':
            self.step_ids = get_step_ids(
                self.step, total_ids, random_seed)
        else:
            self.step_ids = total_ids

    def _load_by_idx(self, idx):
        """Loads image data by idx

        :param idx: the image data idx mapping to a specific data point
        :type idx: int
        :return: a dict containing the loaded image_data
        :rtype: dict
        """

        # extract image data
        image_id = self.step_ids[idx]
        img_dict = self.id_to_img_dict[image_id]
        annos = self.id_to_anno_dict[image_id]
        img_name = img_dict['file_name']
        img_path = os.path.join(self.image_path, img_name)

        # load image
        image = cv2.imread(img_path, 1)

        img_data = dict()
        img_data['image'] = image
        img_data['img_name'] = img_name
        img_data['orig_size'] = img_dict['height'], img_dict['width']
        img_data['image_id'] = img_dict['id']

        if len(annos) > 0:
            targets = np.array(
                [np.array(x['bbox']).flatten() for x in annos])
            categories = np.array([x['category_id'] for x in annos])

            img_data['targets'] = targets
            img_data['categories'] = categories

        return img_data

    def __getitem__(self, idx: int):
        """Loads load and map one image datapoint to the
        formatted requasted by the model input

        :param idx: the image data idx mapping to a specific data point
        :type idx: int
        :return: a dict containing the processed image data
        :rtype: dict
        """

        img_data = self._load_by_idx(idx)
        final_img_data = process_img_data(img_data,
                                            self.annos_encoder,
                                            self.input_size,
                                            self.transformer)


        return final_img_data

    def __len__(self):
        """A function to caclulate the length of the dataset

        :return: the calculated length
        :rtype: int
        """

        return len(self.step_ids)


def _coco_id_to_anno_dict(coco_json_dict):
    """Create a dict mapping each image_id to it's annotations

    :param coco_json_dict: a coco formatted json loaded as dict
    :type coco_json_dict: dict
    :return: the image_id to annotations mapping 
    :rtype: dict
    """

    id_to_anno_dict = dict()

    for image_dict in coco_json_dict['images']:
        id_to_anno_dict[image_dict['id']] = list()

    for anno_dict in coco_json_dict['annotations']:
        if 'image_id' in anno_dict.keys():
            anno_key = 'image_id'
        else:
            anno_key = 'id'

        id_to_anno_dict[anno_dict[anno_key]].append(anno_dict)

    return id_to_anno_dict


def _coco_id_to_image_dict(coco_json_dict):
    """Create a dict mapping each image_id to image data

    :param coco_json_dict: a coco formatted json loaded as dict
    :type coco_json_dict: dict
    :return: the image_id to image data mapping
    :rtype: dict
    """

    id_to_image_dict = dict()
    for image_dict in coco_json_dict['images']:
        id_to_image_dict[image_dict['id']] = image_dict

    return id_to_image_dict


def _parse_coco_annos(annos_path):
    """Parse a coco formated json into an id-to-anno
        and an id-to-image-data mapping

    :param annos_path: path containing the annotation file
    :type annos_path: str
    :return: tuple of the 2 mapping dicts
    :rtype: tuple[dict, dict]
    """

    annos_dict = json_to_dict(annos_path)
    id_to_img_dict = _coco_id_to_image_dict(annos_dict)
    id_to_anno_dict = _coco_id_to_anno_dict(annos_dict)
    return id_to_img_dict, id_to_anno_dict

def process_img_data(img_data,
                     annos_mapper,
                     input_size,
                     transformer):
    """Process image data to make them ready for the model

    :param annos_mapper: maps annotation data to a format ready
        for the model. Can be either a BboxMapper or a MasksMapper
    :type annos_mapper: Union(BboxMapper, MasksMapper)
    :param input_size: tuple containing the input size
        of the model
    :type input_size: tuple[int, int]
    :param transformer: it will apply transforms to an input
        image and annotation data
    :type transformer: albumentations.Compose
    :return: dict containing the processed image data
    :rtype: dict
    """

    image = img_data['image']
    img_height = img_data['orig_size'][0]
    img_width = img_data['orig_size'][1]

    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transformer(image=image, bboxes=img_data['targets'], class_labels=img_data['categories'])
    image, targets, categories = transformed["image"], transformed["bboxes"], transformed["class_labels"]
    
    targets = np.array(
        [[int(x[0]), int(x[1]), int(x[2]), int(x[3])] for x in targets])
    categories = np.array(categories)
    img_data['image'] = image

    cls_gt_box, reg_gt_box = annos_mapper.encode_annos(
        targets, categories)
    img_data['cls_label'] = cls_gt_box
    img_data['bbox_label'] = reg_gt_box

    return img_data


def get_step_ids(step, total_ids, random_seed, val_ratio=0.15):
    """Get datapoint ids for each train procedure step

    :param total_ids: list containing the total dataset ids
    :type total_ids: list[int...]
    :param random_seed: the ransom seed specifying the spliting 
        of the dataset
    :type random_seed: int
    :param val_ratio: specifies the percentage of the training data
        to be used for validation, defaults to 0.15
    :type val_ratio: float, optional
    :return: the chosen datapoint ids
    :rtype: list[int...]
    """

    random.seed(random_seed)
    val_split = int(val_ratio * len(total_ids))
    step_ids = random.sample(total_ids, val_split)

    if step == 'train':
        step_ids = [x for x in total_ids if x not in step_ids]

    return step_ids
