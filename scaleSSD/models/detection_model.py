import pytorch_lightning as pl
import torch

class DetectionModel(pl.LightningModule):
    """This is a class responsible for coordinating the train procedure

    :param model: the model to train
    :type model: nn.Module
    :param input_size: the size of the input expected by the model
    :type input_size: tuple[int, int]
    :param loss: instance of ODLoss, to use for training
    :type loss: ODLoss
    :param lr: initial learning rate for the optimizer
    :type lr: float
    :param epochs: number of training epochs. used for
        lr scheduling
    :type epochs: int
    """

    def __init__(self,
                 model,
                 input_size,
                 loss,
                 lr,
                 epochs):
        """Constructor method
        """

        super().__init__()

        self.test_preds = list()
        
        self.input_size = input_size
        self.loss = loss
        self.lr = lr
        self.epochs = epochs

        self.model = model[0]
        self.bbox_decoder = model[1]
        self.bbox_nms = model[2]

    def forward(self, x, step_tag='test'):
        """Forward input to model output

        :param x: input x
        :type x: torch.Tensor
        :param step_tag: type of step ( train, val, test )
        :type step_tag: str 
        :return: output of the model
        :rtype: sequence of torch.Tensors and lists
        """
        
        model_out = self.model.forward(x)

        if step_tag != 'train':
            model_out = self.bbox_decoder(model_out)

        if step_tag == 'test':
            model_out = self.bbox_nms(model_out)
        
        return model_out

    def training_step(self, batch, batch_idx):
        """Handle the training step

        :param batch: the next batch to input to the model
        :type batch: list
        :param batch_idx: the integer idx of the batch
        :type batch_idx: int
        :return: the calculated train loss
        :rtype: torch.Tensor
        """

        train_loss = self.handle_step(batch, 'train')
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        """Handle the validation step

        :param batch: the next batch to input to the model
        :type batch: list
        :param batch_idx: the integer idx of the batch
        :type batch_idx: int
        :return: the calculated validation loss
        :rtype: torch.Tensor
        """

        val_loss = self.handle_step(batch, 'val')
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Handle the test step

        :param batch: the next batch to input to the model
        :type batch: list
        :param batch_idx: the integer idx of the batch
        :type batch_idx: int
        :return: the calculated test loss
        :rtype: torch.Tensor
        """

        test_loss = self.handle_step(batch, 'test')
        self.log('test_loss', test_loss)
        return test_loss

    def handle_step(self, batch, step_tag):
        """Handle all different steps of the train procedure

        :param batch: the next batch to input to the model
        :type batch: list
        :param step_tag: the tag specifying the step of the 
            train procedure
        :type step_tag: str
        :return: the calculated loss
        :rtype: torch.Tensor
        """

        x = batch['images']
        orig_sizes = batch['orig_sizes']
        img_ids = batch['img_ids']
        annos = batch['annos']
        targets = batch['train_target']
        
        # get preds
        preds = self.forward(x, step_tag)

        loss = self.loss(preds, targets)
        
        self.log(f'{step_tag}_loss', loss)

        if step_tag == 'test':
            batch_bboxes, batch_scores, batch_classes = preds[2:]
            self.update_test_preds(
                img_ids, batch_bboxes, batch_scores, batch_classes, orig_sizes)

        return loss

    def update_test_preds(self, img_ids, batch_bboxes, batch_scores, batch_classes, batch_orig_sizes):
        """Update test predictions. Alter sizes

        :param img_ids: A list of images ids
        :type img_ids: list
        :param batch_bboxes: A list of all the batch boxes 
        :type batch_bboxes: list
        :param batch_scores: A list of all the batch scores
        :type batch_scores: list
        :param batch_classes: A list of all the batch classes
        :type batch_classes: list
        :param batch_orig_sizes: A list of all the batch original sizes
        :type batch_orig_sizes: list
        """

        for bidx in range(len(batch_bboxes)):
            img_id = img_ids[bidx]
            orig_size = batch_orig_sizes[bidx]
            
            bboxes = batch_bboxes[bidx].cpu().numpy()
            bboxes[:, [0, 2]] *= orig_size[1] / self.input_size[1]
            bboxes[:, [1, 3]] *= orig_size[0] / self.input_size[0] 
            
            scores = batch_scores[bidx].cpu().numpy()
            classes = batch_classes[bidx].cpu().numpy()
            
            for idx in range(bboxes.shape[0]):
                bbox = bboxes[idx].tolist()
                score = scores[idx].tolist()
                class_label = classes[idx].tolist()
                anno_dict = dict()
                anno_dict['bbox'] = [int(bbox[0]), int(
                    bbox[1]), int(bbox[2]), int(bbox[3])]
                anno_dict['area'] = anno_dict['bbox'][2] * anno_dict['bbox'][3]
                anno_dict['image_id'] = img_id
                anno_dict['category_id'] = int(class_label)
                anno_dict['score'] = float(score)
                self.test_preds.append(anno_dict)

    def get_test_preds(self):
        """Get the predicted boxes on test dataset

        :return: the predicted boxes
        :rtype: dict
        """

        return self.test_preds

    def configure_optimizers(self):
        """Handle optimizers for training procedure

        :return: the configured optimizer
        :rtype: torchl.optim
        """

        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.001)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[self.epochs / 2, self.epochs - self.epochs / 10],
            gamma=0.1)

        return {'optimizer': optim, 'lr_scheduler': scheduler}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)