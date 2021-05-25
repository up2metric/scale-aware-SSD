import os
import argparse
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import scaleSSD.utils.io_ops as io_ops
from scaleSSD.utils.losses import ODLoss
from scaleSSD.models.detection_model import DetectionModel
from scaleSSD.data.data_modules import DataModule
from scaleSSD.models.custom_ssd import create_detection_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    """Parse arguments from files

    :return: arguments
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, required=True,
                        help="train_images path")
    parser.add_argument('--train_labels', type=str, required=True,
                        help="train_labels path")
    parser.add_argument('--test_images', type=str, required=True,
                        help="test_images path")
    parser.add_argument('--test_labels', type=str, required=True,
                        help="test_labels path")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="specify batch size")
    parser.add_argument('--minibatch_size', type=int, default=10244,
                        help="specify minibatch size after sampling")
    parser.add_argument('--sampler', type=str, default='dns', choices=['dns', 'basic'],
                        help="specify minibatch sampler type")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="specify learning rate")
    parser.add_argument('--epochs', type=int, default=500,
                        help="specify training epochs")
    parser.add_argument('--classes', type=int, default=3,
                        help="specify number of classes for the dataset")
    parser.add_argument('--weights', default=[1,4,3.7],
                        help="specify class weights. must be len(weights) = classes")
    parser.add_argument('--input_size', default=[512,512],
                        help="specify input size for custom ssd model")
    parser.add_argument('--anchors', default=[[25],[38],[58],[87]],
                        help="specify anchors")
    parser.add_argument('--transform_norm_parameters', default=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                        help="specify transform_norm_parameters")
    parser.add_argument('--output', type=str, default='output',
                        help="specify output dir")

    args = parser.parse_args()

    return args


def train(args):
    """Trains a detection model based on a configuration

    :param args: argparse arguments specifying input configuration
        and output folder
    """
    
    # create data paths dict
    data_paths = dict()
    data_paths['train_images'] = args.train_images
    data_paths['train_labels'] = args.train_labels
    data_paths['test_images'] = args.test_images
    data_paths['test_labels'] = args.test_labels

    batch_size = args.batch_size
    minibatch_size = args.minibatch_size
    sampler = args.sampler
    lr = args.lr
    epochs = args.epochs
    no_classes = args.classes
    class_weights = args.weights
    input_size = args.input_size
    anchors = args.anchors
    output = args.output
    transform_norm_parameters = args.transform_norm_parameters


    logger = pl_loggers.TensorBoardLogger(
        '{}/custom_ssd_ckpt/logs'.format(output))


    # add background class
    no_classes = no_classes + 1

    # initialize model
    model, bbox_encoder = create_detection_model(
        anchors, 
        input_size=input_size,
        no_classes=no_classes)

    loss = ODLoss(minibatch_size, sampler, class_weights)

    # initialize detection model
    train_model = DetectionModel(model,
                                   input_size,
                                   loss,
                                   lr,
                                   epochs)

    # create checkpoint-creation callback
    checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          save_top_k=3,
                                          save_last=True,
                                          mode='min')
    
    # initialize trainer
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs,
                         gpus=1,
                         num_sanity_val_steps=0,
                         weights_save_path='{}/custom_ssd_ckpt'.format(output),
                         weights_summary='full',
                         callbacks=[checkpoint_callback])

    # initialize data module
    data_module = DataModule(
        batch_size,
        input_size,
        data_paths,
        no_classes,
        bbox_encoder,
        transform_norm_parameters
        )

    # train model
    trainer.fit(train_model, data_module)

    # test model
    trainer.test(test_dataloaders=data_module.test_dataloader())

    version_path = '{}/custom_ssd_ckpt/default/version_{}'.format(
        output, trainer.logger.version)

    print('Version is: {}'.format(version_path))

    # save all metrics in json
    preds = train_model.get_test_preds()

    io_ops.save_dict(preds, os.path.join(version_path, 'test_preds.json'))
    
    #Ground truth data for coco
    cocoGt = COCO(data_paths['test_labels'])
    #Detection data for coco
    cocoDt = cocoGt.loadRes(os.path.join(version_path, 'test_preds.json'))

    annType = 'bbox'

    imgIds = sorted(cocoGt.getImgIds())

    # coco evaluator
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    train_model.log('mAP IoU=0.50:0.95', round(cocoEval.stats[0] * 1, 2))
    train_model.log('mAP IoU=0.50', round(cocoEval.stats[1] * 1, 2))


if __name__ == '__main__':
    args = parse_args()
    train(args)
