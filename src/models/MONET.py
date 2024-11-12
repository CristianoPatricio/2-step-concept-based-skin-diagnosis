from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from scipy.special import softmax
import numpy as np

class MONET:
    """
    Paper: https://doi.org/10.1038/s41591-024-02887-x
    Model: https://github.com/suinleelab/MONET/tree/main
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        """
        self.processor = AutoProcessor.from_pretrained("chanwkim/monet")  # image_processor, tokenizer
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
        self.model.to("cuda:0")
        self.model.eval()

    def get_concept_reference(self):
        normal_skin=['clean', "smooth", 'Healthy', 'normal', 'soft', 'flat']
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
            'derm7ptconcept_pigment network':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_regression structure':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_pigmentation':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_blue whitish veil':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_vascular structures':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_streaks':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
            'derm7ptconcept_dots and globules':['clean', 'smooth', 'Healthy', 'normal', 'soft', 'flat'],
        }

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

    def calculate_concept_presence_score(
        self,
        image_features_norm,
        prompt_target_embedding_norm,
        prompt_ref_embedding_norm,
        temp=1 / np.exp(4.5944),
    ):
        """
        Calculates the concept presence score based on the given image features and concept embeddings.

        Args:
            image_features_norm (numpy.Tensor): Normalized image features.
            prompt_target_embedding_norm (torch.Tensor): Normalized concept target embedding.
            prompt_ref_embedding_norm (torch.Tensor): Normalized concept reference embedding.
            temp (float, optional): Temperature parameter for softmax. Defaults to 1 / np.exp(4.5944).

        Returns:
            np.array: Concept presence score.
        """

        target_similarity = (
            prompt_target_embedding_norm.float() @ image_features_norm.T.float()
        )
        ref_similarity = prompt_ref_embedding_norm.float() @ image_features_norm.T.float()

        target_similarity_mean = target_similarity.mean(dim=[1])
        ref_similarity_mean = ref_similarity.mean(axis=1)

        concept_presence_score = softmax(
            [target_similarity_mean.numpy() / temp, ref_similarity_mean.numpy() / temp],
            axis=0,
        )[0, :].mean(axis=0)

        return concept_presence_score

    def get_concept_bottleneck(
        self,
        image_features_norm,
        concept_list,
        prompt_info,
        temp = 1 / np.exp(4.5944),
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


    @torch.no_grad()
    def extract_image_features(self, img_batch):
        images = self.processor.image_processor(img_batch, return_tensors="pt")["pixel_values"].to(0)
        image_features = self.model.get_image_features(images)
        return image_features.cpu()
    
    @torch.no_grad()
    def calculate_similarity(self, img_batch, text_batch, img_ids, labels):        
        inputs = self.processor(text=[txt for txt in text_batch], images=img_batch, return_tensors="pt", padding=True).to(0)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)

        return labels[sorted_indices[0][0]], probs[0][sorted_indices[0][0]].cpu().numpy()