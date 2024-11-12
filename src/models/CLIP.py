from transformers import CLIPProcessor, CLIPModel
import torch
from scipy.special import softmax
import numpy as np

class CLIPViTB16:
    """
    Paper: https://arxiv.org/abs/2103.00020
    Model: https://huggingface.co/openai/clip-vit-base-patch16
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", device_map="auto")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        images = self.processor.image_processor(img_batch, return_tensors="pt")["pixel_values"].to(0)
        image_features = self.model.get_image_features(images)
        return image_features.cpu() # NOTE: removed .numpy()
    
    @torch.no_grad()
    def calculate_similarity(self, img_batch, text_batch, img_ids=None, labels=None):        
        inputs = self.processor(text=[txt for txt in text_batch], images=img_batch, return_tensors="pt", padding=True).to(0)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)

        if labels != None:
            return labels[sorted_indices[0][0]], probs[0][sorted_indices[0][0]].cpu().numpy()
        else:
            return probs
    
    def get_prompt_embedding(
        self,
        concept_term_list=[],
        prompt_template_list=[
            "This is skin image of {}",
            "This is dermatology image of {}",
            "This is image of {}",
        ],
        prompt_ref_list=[
            ["This is skin image"],
            ["This is dermatology image"],
            ["This is image"],
        ],
    ):
        """
        Generate prompt embeddings for a concept

        Args:
            concept_term_list (list): List of concept terms that will be used to generate prompt target embeddings.
            prompt_template_list (list): List of prompt templates.
            prompt_ref_list (list): List of reference phrases.

        Returns:
            dict: A dictionary containing the normalized prompt target embeddings and prompt reference embeddings.

        Example usage:
            # For the concept "bullae", we here use the terms "bullae" and "blister" to generate the prompt embedding.
            concept_embedding = get_prompt_embedding(concept_term_list=["bullae", "blister"])
        """
        # target embedding
        prompt_target = [
            [prompt_template.format(term) for term in concept_term_list]
            for prompt_template in prompt_template_list
        ]

        prompt_target_tokenized = [
            self.processor(text=prompt_list, return_tensors="pt", padding=True)["input_ids"] for prompt_list in prompt_target
        ]

        with torch.no_grad():
            prompt_target_embedding = torch.stack(
                [
                    self.model.get_text_features(prompt_tokenized.to(0)).detach().cpu()
                    for prompt_tokenized in prompt_target_tokenized
                ]
            )
        prompt_target_embedding_norm = (
            prompt_target_embedding / prompt_target_embedding.norm(dim=2, keepdim=True)
        )

        # reference embedding
        prompt_ref_tokenized = [
            self.processor(text=prompt_list, return_tensors="pt", padding=True)["input_ids"] for prompt_list in prompt_ref_list
        ]
        with torch.no_grad():
            prompt_ref_embedding = torch.stack(
                [
                    self.model.get_text_features(prompt_tokenized.to(0)).detach().cpu()
                    for prompt_tokenized in prompt_ref_tokenized
                ]
            )
        prompt_ref_embedding_norm = prompt_ref_embedding / prompt_ref_embedding.norm(
            dim=2, keepdim=True
        )

        return {
            "prompt_target_embedding_norm": prompt_target_embedding_norm,
            "prompt_ref_embedding_norm": prompt_ref_embedding_norm,
        } 
    
    def get_concept_bottleneck(
        self,
        image_features_norm,
        concept_list,
        prompt_info,
        temp = 1,
        concept_reference_dict=None
    ):

        x_dict = {}

        for i, concept_target in enumerate(concept_list):

            similarity_list = []

            # sim(img, concept_target)
            similarity_image_target_concept = prompt_info[concept_target]["prompt_target_embedding_norm"].float() @ image_features_norm.T.float()
            similarity_list.append(similarity_image_target_concept.mean(dim=[0,1]).detach().cpu())

            if concept_reference_dict is None:
                # sim(img, ref_template)
                similarity_image_ref_template = prompt_info[concept_target]["prompt_ref_embedding_norm"].float() @ image_features_norm.T.float()
                similarity_list.append(similarity_image_ref_template.mean(dim=[0,1]).detach().cpu())
            else:

                for concept_ref in concept_reference_dict[concept_target]:
                    # sim(img, ref_concept)
                    similarity_image_ref_concept = prompt_info[concept_ref]["prompt_target_embedding_norm"].float() @ image_features_norm.T.float()
                    similarity_list.append(similarity_image_ref_concept.mean(dim=[0,1]).detach().cpu())

            x_dict[concept_target] = np.stack(similarity_list).T

        if concept_reference_dict is not None:
            x_softmax = np.array(
                [softmax(x_dict[concept] / temp, axis=1)[:, 0] for concept in x_dict.keys()]
            ).T
        else:
            x_softmax = np.array(
                [(x_dict[concept] / temp)[:, 0] for concept in x_dict.keys()]
            ).T
        
        return x_softmax.squeeze().tolist()
    
    """ @torch.no_grad()
    def get_concept_bottleneck(self, img_batch, text_batch):
        breakpoint()        
        inputs = self.processor(text=[txt for txt in text_batch], images=img_batch, return_tensors="pt", padding=True).to(0)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = softmax([logits_per_image[:,0].mean().cpu().numpy(), logits_per_image[:,1:].mean().cpu().numpy()]) # we can take the softmax to get the label probabilities

        return probs[0]
 """