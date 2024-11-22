from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

from scipy.special import softmax
import torch
import numpy as np

class BiomedCLIP:
    """
    Paper: https://arxiv.org/abs/2303.00915
    Model: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    Requirements: pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib (Better to use a separate conda env due to the transformers version)
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        """

        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device='cuda')
        self.model.eval()
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.context_length = 256

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        images = torch.stack([self.preprocess(img) for img in img_batch]).to(0)
        image_features = self.model.encode_image(images, normalize=True)
        return image_features.cpu() # NOTE: removed .numpy()
    
    @torch.no_grad()
    def calculate_similarity(self, img_batch, text_batch, img_ids=None, labels=None):
        images = torch.stack([self.preprocess(img) for img in img_batch]).to(0)
        texts = self.tokenizer(text_batch, context_length=self.context_length).to(0)
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)

            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()
            
        """ top_k = -1

        for i, img in enumerate(img_ids):
            pred = labels[sorted_indices[i][0]]

            top_k = len(labels) if top_k == -1 else top_k
            print(img + ':')
            for j in range(top_k):
                jth_index = sorted_indices[i][j]
                print(f'{labels[jth_index]}: {logits[i][jth_index]}')
            print('\n') """

        if labels is not None:
            return labels[sorted_indices[0][0]], logits[0][sorted_indices[0][0]]
        else:
            return logits
    
    @torch.no_grad()
    def get_concept_bottleneck(self, img_batch, text_batch):       
        images = torch.stack([self.preprocess(img) for img in img_batch]).to(0)
        texts = self.tokenizer(text_batch, context_length=self.context_length).to(0)

        image_features, text_features, logit_scale = self.model(images, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach()

        probs = softmax([logits[:,0].mean().cpu().numpy(), logits[:,1:].mean().cpu().numpy()]) # we can take the softmax to get the label probabilities

        return probs[0]
    
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
            self.tokenizer(prompt_list, context_length=self.context_length) for prompt_list in prompt_target
        ]

        with torch.no_grad():
            prompt_target_embedding = torch.stack(
                [
                    self.model.encode_text(prompt_tokenized.to(0)).detach().cpu()
                    for prompt_tokenized in prompt_target_tokenized
                ]
            )
        prompt_target_embedding_norm = (
            prompt_target_embedding / prompt_target_embedding.norm(dim=2, keepdim=True)
        )

        # reference embedding
        prompt_ref_tokenized = [
            self.tokenizer(prompt_list, context_length=self.context_length) for prompt_list in prompt_ref_list
        ]
        with torch.no_grad():
            prompt_ref_embedding = torch.stack(
                [
                    self.model.encode_text(prompt_tokenized.to(0)).detach().cpu()
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
    
    """
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
        """
        

                                      