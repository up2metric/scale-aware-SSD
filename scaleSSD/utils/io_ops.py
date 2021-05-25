import json
import torch
import os

def json_to_dict(json_file):
    """Loads a json to python dictionary

    :param json_file: json file to load
    :type json_file: str
    :return: dictionary of loaded json
    :rtype: dict
    """

    with open(json_file) as f:
        json_dict = json.load(f)
        return json_dict


def save_dict(json_dict, output_file):
    """Saves dictionary to json format

    :param json_dict: dictionary to save
    :type json_dict: dict
    :param output_file: file to save the outputed json
    :type output_file: str
    """

    with open(output_file, 'w') as fp:
        json.dump(json_dict, fp, indent=4, sort_keys=True, default=str)

def get_best_ckpt(ckpts_path):
    """Detects and returns the best pytorch lightning formated checkpoint

    :param ckpts_path: folder path to search best checkpoint
    :type ckpts_path: str
    :return: best checkpoint path
    :rtype: str
    """
    
    print(ckpts_path)

    ckpt_path = ''

    for checkpoint in os.listdir(ckpts_path):
        checkpoint_dict = torch.load(os.path.join(ckpts_path, checkpoint))
        class_key = list(checkpoint_dict['callbacks'].keys())[0]
        callback_dict = checkpoint_dict['callbacks'][class_key]
        print(callback_dict)
        if float(callback_dict['current_score']) == float(callback_dict['best_model_score']):
            ckpt_path = callback_dict['best_model_path']
            break
    print(ckpt_path)

    return ckpt_path