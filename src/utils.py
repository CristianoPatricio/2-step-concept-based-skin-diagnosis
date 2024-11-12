import os
import json
import pandas as pd
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from src.data.PH2_dataset import PH2Dataset
from src.data.HAM10000_dataset import HAM10000Dataset


def seed_everything(seed=0):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set seed for general python operations
    random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # Set the seed for Numpy operations
    np.random.seed(seed)
    # To use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # TO use deterministic benchmark
    torch.backends.cudnn.benchmark = False

def map_letter_to_label(letter:str)->str:
    return 'nevus' if letter == 'A' else 'melanoma'

def map_label_to_name(label:int)->str:
    return 'nevus' if label == 0 else 'melanoma'

def get_current_date():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time in yyyy-mm-dd-hh-mm-ss format
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_date

def calculate_metrics(y_true, y_pred, y_pred_probs=None):

    print("\n")
    print(f"Classification Report:")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=["NEV", "MEL"]))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Confusion Matrix:")
    print(conf_matrix,"\n")

    # Calculate AUC score
    if y_pred_probs != None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")
    else:
        roc_auc = None

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity (recall, true positive rate)
    sensitivity = tp / (tp + fn)
    print(f"Sensitivity (Recall, TPR): {sensitivity:.4f}")

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp)
    print(f"Specificity (TNR): {specificity:.4f}")
    print("\n")

    # Save into JSON
    data = {
        "auc": roc_auc,
        "bacc": balanced_acc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity
    }

    return data

def save_data_to_json(data:dict, model:str, dataset:str, subdir:str, split=None, task=None) -> None:
    # Specify the file path
    if split != None:
        if task != None:
            file_path = f'results/{subdir}/{model}_{dataset}_split_{split}_results_{task}.json'
        else:
            file_path = f'results/{subdir}/{model}_{dataset}_split_{split}_results.json'
    else:
        if task != None:
            file_path = f'results/{subdir}/{model}_{dataset}_results_{task}.json'
        else:
            file_path = f'results/{subdir}/{model}_{dataset}_results.json'

    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save results to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)  # indent=4 makes the file human-readable

    print(f"Results saved to {file_path}")

def save_dict_to_csv(dataset, model, dict_responses, task, split=None):
    # Define path
    if split != None:
        file_path = f"results/model_responses/{dataset}_{model}_split_{split}_responses_{task}.csv"

    else:
        file_path = f"results/model_responses/{dataset}_{model}_responses_{task}.csv"

    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame(dict_responses)
    df.to_csv(file_path, index=False)

    print(f"Results saved to {file_path}")

def load_data(dataset:str, split:int):

    if dataset == "PH2":
        dataset_train = PH2Dataset(csv_file=f"data/PH2/splits/PH2_train_split_{split}.csv", img_extension="jpg", path_to_images="data/PH2/images")
        dataset_test = PH2Dataset(csv_file=f"data/PH2/splits/PH2_test_split_{split}.csv", img_extension="jpg", path_to_images="data/PH2/images")
    elif dataset == "Derm7pt":
        dataset_train = PH2Dataset(csv_file="data/Derm7pt/splits/derm7pt_train.csv", img_extension="jpg", path_to_images="data/Derm7pt/images")
        dataset_test = PH2Dataset(csv_file="data/Derm7pt/splits/derm7pt_test.csv", img_extension="jpg", path_to_images="data/Derm7pt/images")
    elif dataset == "HAM10000":
        metadata_file = "data/HAM10000/splits/metadata_ham10000_gt.csv"
        metadata = pd.read_csv(metadata_file)

        test_set = metadata[metadata['split'] == 'test']
        train = metadata[metadata['split'] == 'train']
        
        # Drop lesion Ids from train set that are also in test set
        train_set = train[~train['lesion_id'].isin(test_set['lesion_id'])]

        dataset_train = HAM10000Dataset(root_dir="data/HAM10000/images", metadata=train_set, img_extension='jpg')
        dataset_test = HAM10000Dataset(root_dir="data/HAM10000/images", metadata=test_set, img_extension='jpg')
    else:
        raise ValueError(f"The dataset {dataset} is not implemented.")

    train_dataloader, test_dataloader = DataLoader(dataset_train, shuffle=False, num_workers=4, batch_size=1), DataLoader(dataset_test, shuffle=False, num_workers=4, batch_size=1)

    return train_dataloader, test_dataloader

def generate_template(label: str, concepts: list) -> str:
    return f"""The lesion is diagnosed as {label}. The presence of {", ".join(item for item in concepts)} are highly suggestive of {label}."""

def convert_numbers_to_concepts(concepts: list, concept_reference_dict):
        return [name for name, concept in zip(list(concept_reference_dict.keys()), concepts) if concept == 1]

def create_explicd_config(gpu_id):

    class Values:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Define your configuration
    config = Values()

    config.gpu = str(gpu_id)
    config.dataset = "isic2018"
    config.model = "explicd"
    config.load = "checkpoints/explicd_best.pth"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
    }

    cls_weight_dict = {
        'isic2018': [1, 0.5, 1.2, 1.3, 1, 2, 2], 
    }
    
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]

    return config