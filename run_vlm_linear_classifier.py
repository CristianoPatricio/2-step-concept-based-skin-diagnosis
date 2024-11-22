import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse

from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.PH2_dataset import PH2Dataset
from src.data.HAM10000_dataset import HAM10000Dataset
from src.utils import map_label_to_name, calculate_metrics, seed_everything, save_data_to_json, get_current_date, load_data

clinical_concepts = [
            'typical pigment network',
            'atypical pigment network',
            'irregular streaks',
            'regular streaks',
            'regular dots and globules',
            'irregular dots and globules',
            'blue-whitish veil',
            'regression structures'
        ]


concept_reference_dict = {
    "Asymmetry": ["Symmetry", "Regular", "Uniform"],
    "Irregular": ["Regular", "Smooth"],
    "Black": ["White", "Creamy", "Colorless", "Unpigmented"],
    "Blue": ["Green", "Red"],
    "White": ["Black", "Colored", "Pigmented"],
    "Brown": ["Pale", "White"],
    "Erosion":["Deposition", "Buildup"],
    "Multiple Colors": ["Single Color", "Unicolor"],
    "Tiny": ["Large", "Big"],
    "Regular": ["Irregular"], 
}

def generate_template(label: str, concepts: list) -> str:
    return f"""The lesion is diagnosed as {label}. The presence of {", ".join(item for item in concepts)} are highly suggestive of {label}."""

def convert_numbers_to_concepts(concepts: list):
        return [name for name, concept in zip(list(concept_reference_dict.keys()), concepts) if concept == 1]

def concepts_to_class_label(model_name=None, dataset=None, split=None):

    # Load data
    train_dataloader, test_dataloader = load_data(dataset=dataset, split=split)

    # Initialize model
    if model_name == "MONET":
        """MONET"""
        from src.models.MONET import MONET 
        model = MONET()
    elif model_name == "CLIP":
        """CLIP"""
        from src.models.CLIP import CLIPViTB16
        model = CLIPViTB16()
    elif model_name == "BiomedCLIP":
        """BiomedCLIP"""
        from src.models.BiomedCLIP import BiomedCLIP
        model = BiomedCLIP()
    elif model_name == "ExpLICD":
        """ExpLICD"""
        from src.models.Explicd import Explicd
        from src.utils import create_explicd_config
        config = create_explicd_config(gpu_id=2) # TODO: Make this dynamically
        model = Explicd(config=config)
    elif model_name == "CBIVLM":
        pass
    else:
        raise ValueError(f"The model {model} has not a valid implementation.")
    
    if model_name == "MONET":

        # Get concept prompts
        prompt_info = {}
        for concept in concept_reference_dict.keys():
            prompt_info[concept] = model.get_prompt_embedding(concept_term_list=[concept])
            for ref_concept in concept_reference_dict[concept]:
                prompt_info[ref_concept] = model.get_prompt_embedding(concept_term_list=[ref_concept])

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            # Get image embedding
            image_embedding = model.extract_image_features(imgs)

            if model_name != "BiomedCLIP":
                image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
            else:
                image_features_norm = image_embedding

            # Get scores
            scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)

            x_train.append(scores)
            y_train.append(y_true.item())
        
        for batch in tqdm(test_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            # Get image embedding
            image_embedding = model.extract_image_features(imgs)

            if model_name != "BiomedCLIP":
                image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
            else:
                image_features_norm = image_embedding

            # Get scores
            scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)

            x_test.append(scores)
            y_test.append(y_true.item())


        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(np.array(x_train), np.array(y_train).squeeze())

        results = calculate_metrics(y_test, y_pred=clf.predict(x_test).tolist(), y_pred_probs=clf.predict_proba(x_test)[:, 1].tolist())
        save_data_to_json(data=results, subdir='vlm_linear_classifier', model=model_name, dataset=dataset, split=split, task="vlm_linear_classifier")

    elif model_name == "CLIP":

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            scores = []
            for concept in concept_reference_dict.keys():
                template = 'this is a dermoscopic image of '
                labels = [concept]
                for ref_concept in concept_reference_dict[concept]:
                    labels.append(ref_concept)
                pred_probs = model.get_concept_bottleneck(img_batch=imgs, text_batch=[template + l for l in labels], img_ids=img_ids, labels=labels)
                scores.append(pred_probs)

            x_train.append(scores)
            y_train.append(y_true.item())
        
        for batch in tqdm(test_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            scores = []
            for concept in concept_reference_dict.keys():
                template = 'this is a dermoscopic image of '
                labels = [concept]
                for ref_concept in concept_reference_dict[concept]:
                    labels.append(ref_concept)
                pred_probs = model.get_concept_bottleneck(img_batch=imgs, text_batch=[template + l for l in labels], img_ids=img_ids, labels=labels)
                scores.append(pred_probs)

            x_test.append(scores)
            y_test.append(y_true.item())


        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(np.array(x_train), np.array(y_train).squeeze())

        results = calculate_metrics(y_test, y_pred=clf.predict(x_test).tolist(), y_pred_probs=clf.predict_proba(x_test)[:, 1].tolist())
        save_data_to_json(data=results, subdir='vlm_linear_classifier', model=model_name, dataset=dataset, split=split, task="vlm_linear_classifier")

    elif model_name == "BiomedCLIP":

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            scores = []
            for concept in concept_reference_dict.keys():
                template = ' presented in image'
                labels = [concept]
                for ref_concept in concept_reference_dict[concept]:
                    labels.append(ref_concept)
                pred_probs = model.get_concept_bottleneck(img_batch=imgs, text_batch=[l + template for l in labels])
                scores.append(pred_probs)

            x_train.append(scores)
            y_train.append(y_true.item())
        
        for batch in tqdm(test_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]
            
            scores = []
            for concept in concept_reference_dict.keys():
                template = ' presented in image'
                labels = [concept]
                for ref_concept in concept_reference_dict[concept]:
                    labels.append(ref_concept)
                pred_probs = model.get_concept_bottleneck(img_batch=imgs, text_batch=[l + template for l in labels])
                scores.append(pred_probs)

            x_test.append(scores)
            y_test.append(y_true.item())


        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(np.array(x_train), np.array(y_train).squeeze())

        results = calculate_metrics(y_test, y_pred=clf.predict(x_test).tolist(), y_pred_probs=clf.predict_proba(x_test)[:, 1].tolist())
        save_data_to_json(data=results, subdir='vlm_linear_classifier', model=model_name, dataset=dataset, split=split, task="vlm_linear_classifier")

    elif model_name == "ExpLICD":

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()

            _, scores = model.get_concept_predictions(batch=batch, config=config)

            x_train.append(scores)
            y_train.append(y_true.item())
        
        for batch in tqdm(test_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            
            _, scores = model.get_concept_predictions(batch=batch, config=config)

            x_test.append(scores)
            y_test.append(y_true.item())

        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(np.array(x_train), np.array(y_train).squeeze())

        results = calculate_metrics(y_test, y_pred=clf.predict(x_test).tolist(), y_pred_probs=clf.predict_proba(x_test)[:, 1].tolist())
        save_data_to_json(data=results, subdir='vlm_linear_classifier', model=model_name, dataset=dataset, split=split, task="vlm_linear_classifier")

    elif model_name == "CBIVLM":

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        data = pd.read_csv(f"results/clip_ft_reports/{dataset}_dermatology_reports_generated_by_CLIP_FT_raw_values_True.csv")
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()

            scores = data[data.image_id == img_ids[0]][[str(i) for i in range(10)]].values.tolist()[0]
           
            x_train.append(scores)
            y_train.append(y_true.item())
        
        for batch in tqdm(test_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            
            scores = data[data.image_id == img_ids[0]][[str(i) for i in range(10)]].values.tolist()[0]

            x_test.append(scores)
            y_test.append(y_true.item())

        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(np.array(x_train), np.array(y_train).squeeze())

        results = calculate_metrics(y_test, y_pred=clf.predict(x_test).tolist(), y_pred_probs=clf.predict_proba(x_test)[:, 1].tolist())
        save_data_to_json(data=results, subdir='vlm_linear_classifier', model=model_name, dataset=dataset, split=split, task="vlm_linear_classifier")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concept to Class label Classification')
    parser.add_argument('--model', type=str, help='Name of the model to evaluate. Available options: (CLIP, BiomedCLIP, MONET, ExpLICD, CBIVLM)', default='CLIP')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate. Available options: (PH2, Derm7pt, HAM10000)', default='Derm7pt')
    parser.add_argument('--split', type=int, help='Split of the dataset if exists. Only in the case of PH2 dataset.', default=None)
    args = parser.parse_args()

    print("\n")
    print("#==============================================================================")
    print(f"# Status:       Running...")
    print(f"# Model:        {args.model}")
    print(f"# Dataset:      {args.dataset}")
    print(f"# Split:        {args.split}")
    print(f"# Date:         {get_current_date()}")
    print("#==============================================================================")

    seed_everything(seed=42)
    concepts_to_class_label(model_name=args.model, dataset=args.dataset, split=args.split)

    print("\n")
    print("#==============================================================================")
    print(f"# Status:       Finished!")
    print(f"# Date:         {get_current_date()}")
    print("#==============================================================================")
    print("\n")

    """
    To get results avg by splits in PH2:
    python calculate_metrics.py --model=MONET --task=PH2_eval --task_model=CBM
    """
