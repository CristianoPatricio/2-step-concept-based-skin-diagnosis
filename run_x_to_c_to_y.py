import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse
import os
import gc
import torch

from src.models.MONET import MONET 
from src.models.CLIP import CLIPViTB16
from src.models.BiomedCLIP import BiomedCLIP
from src.models.MMed_Llama_3_8B import MMedLlama3
from src.models.Explicd import Explicd
from src.models.Mistral import Mistral
from src.models.GPT4o import GPT4o
from src.utils import map_label_to_name, load_data, generate_template, convert_numbers_to_concepts, map_letter_to_label, calculate_metrics, save_data_to_json, seed_everything, get_current_date
from src.rices import RICES

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


concept_reference_dict_MONET = {
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

concept_reference_dict_PH2 = {
    "typical pigment network": ["atypical pigment network", "absence of pigment network"],
    "atypical pigment network": ["typical pigment network", "regular pigment network"],
    "irregular streaks": ["regular streaks"],
    "regular streaks": ["irregular streaks"],
    "regular dots and globules": ["irregular dots and globules"],
    "irregular dots and globules": ["regular dots and globules"],
    "blue-whitish veil": ["clear", "non-whitish skin"],
    "regression structures": ["progression structures"]
}

concept_reference_dict_HAM10000 = {
    "thick reticular or branched lines": ["thin", "straight lines"],
    "black dots or globules in the periphery of the lesion": ["Absence of dots or globules in the center of the lesion"],
    "white lines or white structureless area": ["Dark lines or dark structured areas"],
    "eccentrically located structureless area": ["Centrally located structured area"],
    "grey patterns": ["Bright or colorful patterns"],
    "polymorphous vessels": ["Monomorphic vessels"],
    "pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference": ["Smooth or uniform margin with a complete lesional circumference"],
    "asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria": ["Symmetric combination of a single pattern or color with the presence of other melanoma criteria"],
    "melanoma simulator": ["Benign skin condition analyzer"],
}


def x_to_c(model_name: str, dataset: str, concept_reference_dict: str, split: int=None, raw_values: bool=False, predict_for_train_set: bool=False) -> None:
    """Predicts concepts from MONET.

    Args:
        model_name (str): Name of the model. 
        dataset (str): Name of the dataset.
        concept_reference_dict (dict): Dictionary containing clinical concepts.
        split (int, optional): Split to use in the case of PH2. Defaults to None.
        raw_values (bool, optional): Whether to use raw concepts or not. Defaults to False.
        predict_for_train_set (bool, optional): Whether to generate the reports for training set.

    Returns:
        None: Save predicted concepts into a CSV file.
    """

    if concept_reference_dict == "PH2":
        concept_reference_dict = concept_reference_dict_PH2
    else:
        concept_reference_dict = concept_reference_dict_HAM10000

    if model_name == "MONET":
        concept_reference_dict = concept_reference_dict_MONET

    # Load data
    train_dataloader, test_dataloader = load_data(dataset=dataset, split=split)

    # Initialize model
    if model_name == "MONET":
        model = MONET()
    elif model_name == "CLIP":
        model = CLIPViTB16()
    elif model_name == "BiomedCLIP":
        model = BiomedCLIP()
    elif model_name == "Explicd":
        from src.utils import create_explicd_config
        config = create_explicd_config(gpu_id=2)    # TODO: Make this dynamically
        model = Explicd(config=config)
    else:
        raise TypeError(f"The specififed model {model_name} does not have a valid implementation.")

    # Get concept prompts
    if model_name == "MONET":
        prompt_info = {}
        for concept in concept_reference_dict.keys():
            prompt_info[concept] = model.get_prompt_embedding(concept_term_list=[concept])
            for ref_concept in concept_reference_dict[concept]:
                prompt_info[ref_concept] = model.get_prompt_embedding(concept_term_list=[ref_concept])

    dict_to_save_data = dict()
    for batch in tqdm(test_dataloader):
        img_ids = batch["img_id"]
        y_true = batch["class_label"].numpy()
        imgs = [Image.open(x) for x in batch["img_path"]]

        if model_name != "Explicd":
            image_embedding = model.extract_image_features(imgs)
            image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)

        if model_name == "MONET":
            scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)
        elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
            scores = []
            for concept in concept_reference_dict.keys():
                prompt_template = "{} pattern presented in image"
                text_input = []
                text_input.extend([prompt_template.format(concept)])
                text_input.extend([prompt_template.format(term) for term in concept_reference_dict[concept]])
                scores.append(model.calculate_similarity(img_batch=imgs, text_batch=text_input))
        elif model_name == "Explicd":
            predicted_concepts, _ = model.get_concept_predictions(batch=batch, config=config) 

        if not raw_values:
            if model_name == "MONET":
                mapped_values = list(map(lambda x: 1 if x > 0.5 else 0, scores))

                # Get dermoscopic concepts
                clinical_concepts = convert_numbers_to_concepts(mapped_values, concept_reference_dict=concept_reference_dict)

                # Create template report
                report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
            elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                mapped_values = []
                for val in scores:
                    mapped_values.append(val.argmax())

                # Get dermoscopic concepts
                clinical_concepts = convert_numbers_to_concepts(mapped_values)

                # Create template report
                report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
            elif model_name == "Explicd":
                report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
        else:
            if model_name == "MONET":
                report_template = {
                    "Asymmetry": scores[0],
                    "Irregular": scores[1],
                    "Black": scores[2],
                    "Blue": scores[3],
                    "White": scores[4],
                    "Brown": scores[5],
                    "Erosion": scores[6],
                    "Multiple Colors": scores[7],
                    "Tiny": scores[8],
                    "Regular": scores[9], 
                }
            else:
                report_template = {
                    "Asymmetry": scores[0][:,:1],
                    "Irregular": scores[1][:,:1],
                    "Black": scores[2][:,:1],
                    "Blue": scores[3][:,:1],
                    "White": scores[4][:,:1],
                    "Brown": scores[5][:,:1],
                    "Erosion": scores[6][:,:1],
                    "Multiple Colors": scores[7][:,:1],
                    "Tiny": scores[8][:,:1],
                    "Regular": scores[9][:,:1], 
                }

        dict_to_save_data[img_ids[0]] = str(report_template)
    
    if predict_for_train_set:
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]

            if model_name != "Explicd":
                image_embedding = model.extract_image_features(imgs)
                image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)

            if model_name == "MONET":
                scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)
            elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                scores = []
                for concept in concept_reference_dict.keys():
                    prompt_template = "{} pattern presented in image"
                    text_input = []
                    text_input.extend([prompt_template.format(concept)])
                    text_input.extend([prompt_template.format(term) for term in concept_reference_dict[concept]])
                    scores.append(model.calculate_similarity(img_batch=imgs, text_batch=text_input))
            elif model_name == "Explicd":
                predicted_concepts, _ = model.get_concept_predictions(batch=batch, config=config) 

            if not raw_values:
                if model_name == "MONET":
                    mapped_values = list(map(lambda x: 1 if x > 0.5 else 0, scores))

                    # Get dermoscopic concepts
                    clinical_concepts = convert_numbers_to_concepts(mapped_values, concept_reference_dict=concept_reference_dict)

                    # Create template report
                    report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
                elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                    mapped_values = []
                    for val in scores:
                        mapped_values.append(val.argmax())

                    # Get dermoscopic concepts
                    clinical_concepts = convert_numbers_to_concepts(mapped_values)

                    # Create template report
                    report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
                elif model_name == "Explicd":
                    report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
            else:
                if model_name == "MONET":
                    report_template = {
                        "Asymmetry": scores[0],
                        "Irregular": scores[1],
                        "Black": scores[2],
                        "Blue": scores[3],
                        "White": scores[4],
                        "Brown": scores[5],
                        "Erosion": scores[6],
                        "Multiple Colors": scores[7],
                        "Tiny": scores[8],
                        "Regular": scores[9], 
                    }
                else:
                    report_template = {
                        "Asymmetry": scores[0][:,:1],
                        "Irregular": scores[1][:,:1],
                        "Black": scores[2][:,:1],
                        "Blue": scores[3][:,:1],
                        "White": scores[4][:,:1],
                        "Brown": scores[5][:,:1],
                        "Erosion": scores[6][:,:1],
                        "Multiple Colors": scores[7][:,:1],
                        "Tiny": scores[8][:,:1],
                        "Regular": scores[9][:,:1], 
                    }

            dict_to_save_data[img_ids[0]] = str(report_template)

    # Save reports into CSV file
    pre_df = pd.DataFrame.from_dict(dict_to_save_data, orient='index', columns=['Column 2']).reset_index()
    pre_df.columns = ["image_id", "report"]
    if split is None:
        file_path = f"results/concept_prediction/{dataset}_dermatology_reports_generated_by_{model_name}_raw_values_{raw_values}.csv"
    else:
        file_path = f"results/concept_prediction/{dataset}_split_{split}_dermatology_reports_generated_by_{model_name}_raw_values_{raw_values}.csv"

    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    pre_df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")

    # free GPU memory
    del model
    del test_dataloader
    del dict_to_save_data
    gc.collect()
    torch.cuda.empty_cache()

def c_to_y(model_name: str, dataset:str, ckpt:str, split=None, raw_values=False, concept_extractor:str=None, report_path: str = None, use_demos=False, n_demos=0, ground_truth_concepts=False):
    """
    Report template:
    > The lesion is diagnosed as {label}. The presence of {", ".join(item for item in concepts)} are highly suggestive of {label}.
    """

    # Load reports
    if dataset == 'PH2':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:  
            df_reports = pd.read_csv(f"data/concept_prediction/PH2_split_{split}_dermatology_reports_generated_by_{concept_extractor}_raw_values_{raw_values}.csv")
        
        PH2_TEST = pd.read_csv(f"data/PH2/splits/PH2_test_split_{split}.csv")
        PH2_TRAIN = pd.read_csv(f"data/PH2/splits/PH2_train_split_{split}.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(PH2_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(PH2_TRAIN.images.to_list())]
    elif dataset == 'Derm7pt':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"data/concept_prediction/Derm7pt_dermatology_reports_generated_by_{concept_extractor}_raw_values_{raw_values}.csv")
        
        D7_TEST = pd.read_csv("data/Derm7pt/splits/derm7pt_test.csv")
        D7_TRAIN = pd.read_csv("data/Derm7pt/splits/derm7pt_train.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(D7_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(D7_TRAIN.images.to_list())]
    elif dataset == 'HAM10000':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"data/concept_prediction/HAM10000_dermatology_reports_generated_by_{concept_extractor}_raw_values_{raw_values}.csv")

        HAM_TEST = pd.read_csv("data/HAM10000/splits/HAM10000_test.csv")
        HAM_TRAIN = pd.read_csv("data/HAM10000/splits/HAM10000_train.csv")
        HAM_VAL = pd.read_csv("data/HAM10000/splits/HAM10000_val.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(HAM_TEST.image_id.to_list())]
        df_reports_train = pd.concat([df_reports.loc[df_reports.image_id.isin(HAM_TRAIN.image_id.to_list())], df_reports.loc[df_reports.image_id.isin(HAM_VAL.image_id.to_list())]])
    else:
        raise ValueError(f"The dataset {dataset} is not implemented.")

    # Evaluate
    if model_name == "MMed":
        model = MMedLlama3(ckpt)
    elif model_name == "Mistral":
        model = Mistral()
    elif model_name == "GPT":
        model = GPT4o(model=ckpt)
    else:
        raise TypeError(f"The specififed model {model_name} does not have a valid implementation.")

    dict_responses = {
        'image_id': [],
        'gt_response': [],
        'llm_response': []
    }

    # Define instruction and query
    instruction = f"You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    hint = """Consider the following examples:\n
    A skin lesion is a nevus when it has the majority of the following concepts: uniformly tan, brown, or black, round, sharp and well-defined, regular pigment network, symmetric dots and globules, smooth, symmetrical, raised with possible central ulceration.\n
    A skin lesion is a melanoma when it has the majority of the following concepts: highly variable, often with multiple colors (black, brown, red, white, blue), irregular, often blurry and irregular, atypical pigment network, irregular streaks, blue-whitish veil, irregular, a raised or ulcerated surface, asymmetrical, flat to raised.\n"""
    query = """###Question: What is the type of skin lesion that is associated with the following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""

    # Demonstrations
    if use_demos:
        rices = RICES(dataset=dataset, split=split, valid_ids=[])

    for img_id, report in tqdm(zip(df_reports_test.image_id.to_list(), df_reports_test.report.to_list())):

        # Demonstrations
        if use_demos:
            # Get most similar N image ids to the query image
            demos_ids = rices.get_context_keys(key=img_id, n=n_demos)
            demos_to_use_in_prompt = []
            # Iterate over retrieved demo_ids and save the respective report into a list
            for id in demos_ids:
                sample = df_reports_train[df_reports_train.image_id == id].report.to_list()
                demos_to_use_in_prompt.append(sample[0])
        else:
            demos_to_use_in_prompt = None
    
        if concept_extractor != "Explicd":
            concepts = report[report.find("The presence"):report.find("are highly")-1]
            input_query = query.format(concepts)
            gt_response = report[len("The lesion is diagnosed as "):report.find(". The")]
        else:
            concepts = report[:report.find("Thus the diagnosis is")-1]
            input_query = query.format(concepts)
            gt_response = report[report.find("Thus the diagnosis is ")+len("Thus the diagnosis is "):-1]
        
        if model_name == "GPT": # TODO: maybe uniformize in the future
            """ GPT-4 """
            llm_response = map_letter_to_label(model.inference_text(instruction=instruction, query=input_query, max_new_tokens=1).strip())
        else:
            prompt = model.get_prompt(instruction, input_query, demos=demos_to_use_in_prompt)
            llm_response = map_letter_to_label(model.predict(prompt, max_new_tokens=1).strip())
            
        dict_responses['image_id'].append(img_id)
        dict_responses['gt_response'].append(gt_response)
        dict_responses['llm_response'].append(llm_response)

    # Converter para DataFrame
    df = pd.DataFrame(dict_responses)
    
    if model_name == "MMed":
        if split != None:
            file_path = f"results/label_prediction/{dataset}_split_{split}_{ckpt[ckpt.find('/')+1:]}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv"
        else:
            file_path = f"results/label_prediction/{dataset}_{ckpt[ckpt.find('/')+1:]}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv"
    elif model_name in ["Mistral", "GPT"]:
        if split != None:
            file_path = f"results/label_prediction/{dataset}_split_{split}_{model_name}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv"
        else:
            file_path = f"results/label_prediction/{dataset}_{model_name}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv"
    else:
        raise ValueError("Not found")
    
    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def classification(model_name:str, dataset: str, ckpt: str, split=None, ground_truth_concepts=False, raw_values=False, concept_extractor: str=None, n_demos=0):

    if model_name == "MMed":
        if split != None:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_split_{split}_{ckpt[ckpt.find('/')+1:]}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv")
        else:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_{ckpt[ckpt.find('/')+1:]}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv")
    elif model_name in ["Mistral", "GPT"]:
        if split != None:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_split_{split}_{model_name}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv")
        else:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_{model_name}_diagnostic_report_validation_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_model_extractor_{concept_extractor}_n_demos_{n_demos}.csv")
    else:
        raise ValueError("File not found!")

    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"data/PH2/splits/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("data/Derm7pt/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]
    elif dataset == "HAM10000":
        HAM_TEST = pd.read_csv("data/HAM10000/splits/HAM10000_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(HAM_TEST.image_id.to_list())]

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }
    
    y_true = df_filtered.gt_response.map(mapping).to_list()
    y_pred = df_filtered.llm_response.map(mapping).to_list()
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model_name, subdir='x_to_c_to_y', dataset=dataset, split=split, task=f"gt_concepts_{ground_truth_concepts}_raw_values_{raw_values}_model_extractor_{concept_extractor}_n_demos_{n_demos}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concept to Class label Classification')
    parser.add_argument('--model', type=str, help='Name of the model to evaluate. Choose between (MONET, CLIP, BiomedCLIP, Explicd).', default='CLIP')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', default='Derm7pt')
    parser.add_argument('--report_path', type=str, help='Path to report', default=None)
    parser.add_argument('--split', type=int, help='Split of the dataset if exists', default=None)
    parser.add_argument('--raw_values', action="store_true", help='Include this parameter to save concepts along its concept presence score.')
    parser.add_argument('--ckpt', type=str, help='Name of the model checkpoint', default='Henrychur/MMed-Llama-3-8B')
    parser.add_argument('--concept_extractor', type=str, help='Name of the model used to extract the concepts', default='MONET')
    parser.add_argument('--concept_reference_dict', type=str, help='Name of the model used to extract the concepts', default='PH2')
    parser.add_argument('--llm', type=str, help='Name of the LLM used to provide the final diagnosis. Choose between (MMed, Mistral)', default='MMed')
    parser.add_argument('--use_demos', action="store_true", help='Add this argument if few-shot learning')
    parser.add_argument('--predict_for_train_set', action="store_true", help='Add this argument if you want to generate reports also for training set.')
    parser.add_argument('--n_demos', type=int, help='Number of demos. Valid when --use_demos is added to the config.', default=0)
    parser.add_argument('--gt_concepts', action="store_true", help='Whether or not use gt concepts in the setting c -> y')
    args = parser.parse_args()

    seed_everything(seed=42)

    print("\n")
    print("#==============================================================================")
    print(f"# Status:    Running...")
    print(f"# LLM:       {args.llm}")
    print(f"# Dataset:   {args.dataset}")
    print(f"# n-shots:   {args.n_demos}")
    print(f"# Date:      {get_current_date()}")
    print("#==============================================================================")

    # Uncomment line below to generate the concepts from input images, otherwise, use the provided concepts at 'data/concept_prediction'
    # x_to_c(model_name=args.model, dataset=args.dataset, concept_reference_dict=args.concept_reference_dict, split=args.split, raw_values=args.raw_values, predict_for_train_set=args.predict_for_train_set)
    c_to_y(model_name=args.llm, dataset=args.dataset, ckpt=args.ckpt, split=args.split, raw_values=args.raw_values, concept_extractor=args.concept_extractor, report_path=args.report_path, use_demos=args.use_demos, n_demos=args.n_demos, ground_truth_concepts=args.gt_concepts)
    classification(model_name=args.llm, dataset=args.dataset, ckpt=args.ckpt, split=args.split, ground_truth_concepts=args.gt_concepts,  raw_values=args.raw_values, concept_extractor=args.concept_extractor, n_demos=args.n_demos)

    print("\n")
    print("#==============================================================================")
    print(f"# Status:     Finished!")
    print(f"# Date:       {get_current_date()}")
    print("#==============================================================================")
    print("\n")