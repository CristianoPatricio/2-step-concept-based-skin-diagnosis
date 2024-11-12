import json
import numpy as np
import pandas as pd
import argparse
import os

from src.utils import calculate_metrics, save_data_to_json, get_current_date

def read_json_data(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data

def get_results_ph2(model:str, task_model:str, subdir: str):

    # Calculate results for PH2
    accumulated_bacc = []
    accumulated_sens = []
    accumulated_spec = []
    for split in range(5):
        if task_model != None:
            results = read_json_data(file_path=f"results/{subdir}/{model}_PH2_split_{split}_results_{task_model}.json")
        else:
            results = read_json_data(file_path=f"results/{subdir}/{model}_PH2_split_{split}_results.json")
        accumulated_bacc.append(results["bacc"])
        accumulated_sens.append(results["sensitivity"])
        accumulated_spec.append(results["specificity"])

    print("\n")
    print(f"Mean BACC: {np.array(accumulated_bacc).mean():.4f}")
    print(f"Mean Sensitivity: {np.array(accumulated_sens).mean():.4f}")
    print(f"Mean Specificity: {np.array(accumulated_spec).mean():.4f}")

def get_results_after_extracting_labels_with_mmed(model:str, dataset:str, ckpt:str, split=None, task=None):
    if split != None:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_{model}_split_{split}_responses_extracted_by_MMed.csv")
    else:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_{model}_responses_extracted_by_MMed.csv")

    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"/home/jcneves/CBM/concept-based-interpretability-VLM/data/PH2/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("/home/jcneves/multimodal-LLM-explainability-dev/data/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]

    elif dataset == "HAM10000":
        df_filtered = df_responses

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }
    
    y_true = df_filtered.gt_response.map(mapping).to_list()
    y_pred = df_filtered.llm_response.map(mapping).to_list()
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model, dataset=dataset, split=split, task=task)

def get_results_after_prediting_labels_with_gpt4o(model:str, dataset:str, ckpt:str, split=None, ground_truth_concepts=False, raw_values=False):
    if split != None:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_split_{split}_{model}_diagnostic_report_validation_gt_concepts_{ground_truth_concepts}_raw_values_{raw_values}.csv")
    else:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_{model}_diagnostic_report_validation_gt_concepts_{ground_truth_concepts}_raw_values_{raw_values}.csv")

    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"/home/jcneves/CBM/concept-based-interpretability-VLM/data/PH2/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("/home/jcneves/multimodal-LLM-explainability-dev/data/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]

    elif dataset == "HAM10000":
        df_filtered = df_responses

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }

    y_true = []
    y_pred = []
    for img in df_filtered["image_id"].to_list():
        y_true.append(PH2_TEST[PH2_TEST["images"] == img].labels.item())

        # Convert JSON string to dictionary
        result_dict = json.loads(df_filtered[df_filtered.image_id == img]["llm_response"].item())
        y_pred.append(result_dict["answer"].lower())

    y_pred = [mapping[label] for label in y_pred]
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model, dataset=dataset, split=split, task="concepts_to_label_with_image")

def get_results_gpt4o_zero_shot(model:str, dataset:str, split=None):
    if split != None:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_split_{split}_{model}_zero_shot.csv")
    else:
        df_responses = pd.read_csv(f"results/model_responses/{dataset}_{model}_zero_shot.csv")

    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"/home/jcneves/CBM/concept-based-interpretability-VLM/data/PH2/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("/home/jcneves/multimodal-LLM-explainability-dev/data/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]

    elif dataset == "HAM10000":
        df_filtered = df_responses

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }

    y_true = []
    y_pred = []
    for img in df_filtered["image_id"].to_list():

        if dataset == "PH2":
            y_true.append(df_filtered[df_filtered["images"] == img].labels.item())
        elif dataset == "Derm7pt":
            y_true.append(D7_TEST[D7_TEST["images"] == img].labels.item())
        elif dataset == "HAM10000":
            metadata_file = "/home/jcneves/CBM/validating-cbe-via-llm/data/metadata_ham10000_gt.csv"
            metadata = pd.read_csv(metadata_file)

            test_set = metadata[metadata['split'] == 'test']
            y_true.append(test_set[test_set.image_id == img].benign_malignant.item())

        # Convert JSON string to dictionary
        result_dict = json.loads(df_filtered[df_filtered.image_id == img]["gpt_response"].item())
        y_pred.append(result_dict["answer"].lower())

    y_pred = [mapping[label] for label in y_pred]
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model, dataset=dataset, split=split, task="zero_shot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Metrics')
    parser.add_argument('--model', type=str, help='Name of the model', default='CLIP')
    parser.add_argument('--subdir', type=str, help='Subdir', default='0-shot')
    parser.add_argument('--ckpt', type=str, help='Name of the model checkpoint', default='MMed-Llama-3-8B')
    parser.add_argument('--task', type=str, help='Name of the task', default='PH2_eval')
    parser.add_argument('--task_model', type=str, help='Name of the task (open-ended)', default=None)
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', default='Derm7pt')
    parser.add_argument('--split', type=int, help='Split of the dataset if exists', default=None)
    parser.add_argument('--file_path', type=int, help='File path for PH2 eval', default=None)
    parser.add_argument('--gt_concepts', action="store_true", help='Whether or not use gt concepts')
    parser.add_argument('--raw_values', action="store_true", help='Whether or not use raw values')
    args = parser.parse_args()

    print("\n")
    print("#==============================================================================")
    print(f"# Status:           Running...")
    print(f"# Model:            {args.model}")
    print(f"# Date:             {get_current_date()}")
    print("#==============================================================================")

    if str(args.task).__contains__("PH2_eval"):
        get_results_ph2(model=args.model, subdir=args.subdir, task_model=args.task_model)
    elif str(args.task).__contains__("GPT4o_c_to_y"):
        get_results_after_prediting_labels_with_gpt4o(model=args.model, dataset=args.dataset, ckpt=args.ckpt, split=args.split, ground_truth_concepts=args.gt_concepts, raw_values=args.raw_values)
    elif str(args.task).__contains__("GPT4o_zero_shot"):
        get_results_gpt4o_zero_shot(model=args.model, dataset=args.dataset, split=args.split)

    print("\n")
    print("#==============================================================================")
    print(f"# Status:  Finished!")
    print(f"# Date:    {get_current_date()}")
    print("#==============================================================================")
    print("\n")
