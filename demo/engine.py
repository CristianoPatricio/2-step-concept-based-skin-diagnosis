import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import gc
import torch

# Get the absolute path of the '../' directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# Add the directory to the system path
sys.path.append(models_dir)

from src.models.MMed_Llama_3_8B import MMedLlama3
from src.models.Explicd import Explicd
from src.utils import map_label_to_name, load_data, generate_template, convert_numbers_to_concepts, map_letter_to_label, calculate_metrics, save_data_to_json, seed_everything, get_current_date, create_explicd_config, seed_everything
from src.rices import RICES

def x_to_c(pil_image, model_step_1) -> None:
    """Predicts concepts from MONET.

    Args:
        pil_image (PIL image): PIL image from gradio upload image component.

    Returns:
        predicted_concepts: Predicted concepts by ExpLICD.
    """

    predicted_concepts, raw_scores = model_step_1.get_concept_predictions_for_a_single_image(pil_image=pil_image) 

    # free GPU memory
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()

    return predicted_concepts, raw_scores

def c_to_y(intermediate_prompt, model_step_2, use_demos=False, n_demos=0):    
    """Generates final diagnosis grounded on the predicted concepts
    
    Args:
        intermediate_prompt (str): Prompt to ask LLM.

    Returns:
        llm_response: LLM response.
    """
    
    # Define instruction and query
    instruction = intermediate_prompt[:intermediate_prompt.find("###Question:")]
    input_query = intermediate_prompt[intermediate_prompt.find("###Question:"):]

    prompt = model_step_2.get_prompt(instruction, input_query, demos=None)
    llm_response = map_letter_to_label(model_step_2.predict(prompt, max_new_tokens=1).strip())
            
    return llm_response


def load_models():
    seed_everything(seed=42)
    
    # Load ExpLICD
    config = create_explicd_config(gpu_id=0)    # TODO: Make this dynamically
    model_step_1 = Explicd(config=config)

    # Load MMedLlama3
    model_step_2 = MMedLlama3(ckpt="Henrychur/MMed-Llama-3-8B-EnIns")

    return model_step_1, model_step_2
    

