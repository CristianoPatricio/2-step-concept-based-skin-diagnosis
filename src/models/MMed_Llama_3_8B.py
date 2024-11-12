import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MMedLlama3:

    def __init__(self, ckpt="Henrychur/MMed-Llama-3-8B") -> None:
        """
        Initialize the attributes of the class.

        Paper: https://arxiv.org/abs/2402.13963
        Model: https://huggingface.co/Henrychur/
        
        Args:
            ckpt (str): The name of the checkpoints [Henrychur/MMed-Llama-3-8B, Henrychur/MMed-Llama-3-8B-EnIns]
        """

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=2048, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
        self.model.cuda()

    def get_prompt(self, instruction, query, demos=None):
        """Creates the prompt.

        Args:
            diagnosis (str): A set of dermoscopic concepts.

        Returns:
            str: The complete prompt.
        """
        #lang = "english"
        #question = "What is the type of skin lesion that is associated with the following dermoscopic concepts: presence of atypical pigment network, irregular streaks, irregular dots and globules, blue-whitish veil."
        #options = "A. Nevus\nB. Melanoma"

        #tmp = {
        #    #"instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the given diagnosis in {lang}. Answer with the best option directly.",
        #    "instruction": f"You're a {lang} doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations.",
        #    "input": f"###Question: What is the type of skin lesion that is associated with the following dermoscopic concepts: {diagnosis}. ###Options: A. Nevus\nB. Melanoma. ###Answer:",
        #}


        if demos is not None:
            demonstration = "Consider the following examples:\n"
            for d in demos:
                demonstration += f"{d}\n"
                
            tmp = {
                "instruction": instruction,
                "demonstration": demonstration,
                "input": query
            }  

            prompt = (tmp["instruction"] + tmp["demonstration"] + tmp["input"])
        
        else:
            tmp = {
                "instruction": instruction,
                "input": query
            }  

            prompt = (tmp["instruction"] + tmp["input"])
        
        return prompt

    def get_length_of_prompt(self, original_prompt: str) -> str:
        return len(original_prompt)
        
    def predict(self, prompt, max_new_tokens):
        #input_str = ("You are a {language} doctor. Analyze the reasons behind choosing this particular option in 100 words for the following question in {language}. ###Question: {question} ###Options: {options} ###Answer: {answer_id} ###Rationale:")

        model_inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False,
        )

        with torch.no_grad():
            topk_output = self.model.generate(
                model_inputs.input_ids.cuda(),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                top_k=50
            )

        output_str = self.tokenizer.batch_decode(topk_output)  # a list containing just one str

        return output_str[0][self.get_length_of_prompt(prompt):]


