
import torch
import random
import pytorch_lightning as pl
from scaleSSD.data.coco_dataset import COCODataset


class DataModule(pl.LightningDataModule):
    """This is a class, responsible for initialing 
    the datasets for each step

    :param batch_size: the max size of each batch
    :type batch_size: int
    :param input_size: tuple containing the input size
        of the model
    :type input_size: tuple[int, int]
    :param data_paths: dict containing paths for train 
        and test datasets
    :type data_paths: dict
    :param no_classes: number of classes in the dataset
    :type no_classes: int
    :param annos_encoder: instance of BboxEncoder, responsible
        for encoding bounding box annotations
    :type annos_encoder: BboxEncoder
    :param transform_norm_parameters: Parameters for normalization in transforms
    :type transform_norm_parameters: list
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 data_paths,
                 no_classes,
                 annos_encoder,
                 transform_norm_parameters):
        """Constructor method
        """

        super().__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.data_paths = data_paths
        self.annos_encoder = annos_encoder
        self.no_classes = no_classes
        self.transform_norm_parameters = transform_norm_parameters

    
    def get_dataset(self, step, rseed=0):
        """Create and return a dataset for each step (train, val, test)

        :param step: the the of the train procedure. Can be (train, test, val)
        :type step: str
        :param rseed: the random seed to initialize dataset spliting,
            defaults to 0
        :type rseed: int, optional
        :return: the created dataset
        :rtype: torch.utils.data.Dataset
        """

        # if dataset not specified return false
        if step == 'test':    
            images_path = self.data_paths['test_images']
            annos_path = self.data_paths['test_labels']
        else:
            images_path = self.data_paths['train_images']
            annos_path = self.data_paths['train_labels']     
                   
        dataset = COCODataset(step,
                              annos_path,
                              images_path,
                              no_classes=self.no_classes,
                              input_size=self.input_size,
                              annos_encoder=self.annos_encoder,
                              random_seed=rseed,
                              transform_norm_parameters=self.transform_norm_parameters
                            )

        return dataset

    def prepare_data(self):
        """Function to create and prepare datasets
        """

        rand_seed = random.randint(0, 100)

        self.train_dataset = self.get_dataset(
            'train', rand_seed)
        self.val_dataset = self.get_dataset(
            'val', rand_seed)
        self.test_dataset = self.get_dataset(
            'test')

    def train_dataloader(self):
        """Create the train dataloader

        :return: the train dataloader
        :rtype: torch.utils.data.Dataloader
        """

        if self.train_dataset:
            train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=0,
                                                       shuffle=True,
                                                       collate_fn=self.my_collate)
            return train_loader

    def val_dataloader(self):
        """Create the val dataloader

        :return: the val dataloader
        :rtype: torch.utils.data.Dataloader
        """

        if self.val_dataset:
            val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                     batch_size=self.batch_size,
                                                     num_workers=0,
                                                     shuffle=False,
                                                     collate_fn=self.my_collate)
            return val_loader

    def test_dataloader(self):
        """Create the test dataloader

        :return: the test dataloader
        :rtype: torch.utils.data.Dataloader
        """

        if self.test_dataset:
            test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=self.batch_size,
                                                      num_workers=0,
                                                      shuffle=False,
                                                      collate_fn=self.my_collate)
            return test_loader

    def my_collate(self, batch):
        """A custom collate function to group batch data

        :param batch: a list of dicts containing the batch elements
        :type batch: list[dict...]
        :return: a dict containing the grouped batch data
        :rtype: dict
        """

        return_dict = dict()
        return_dict['images'] = torch.stack([item['image'] for item in batch])
        return_dict['img_names'] = [item['img_name'] for item in batch]
        return_dict['orig_sizes'] = [item['orig_size'] for item in batch]
        return_dict['img_ids'] = [item['image_id'] for item in batch]

        targets = [item['targets'] for item in batch]
        categories = [item['categories'] for item in batch]
        return_dict['annos'] = (targets, categories)

        cls_target = torch.stack([item['cls_label'] for item in batch])
        reg_target = torch.stack([item['bbox_label'] for item in batch])
        return_dict['train_target'] = (cls_target, reg_target)

        return return_dict
