from openai import OpenAI
import base64

class GPT4o:

    def __init__(self, model) -> None:
        self.client = OpenAI()
        self.model = model

    def encode_image(self, image_path):
        # Open the image file and encode it as a base64 string
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def inference_text(self, instruction, query, max_new_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=max_new_tokens  # Limit the output to N tokens
        )

        return completion.choices[0].message.content
    
    def inference_vision(self, instruction, query, base64_image, max_new_tokens):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "low"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=max_new_tokens  # Limit the output to N tokens
        )

        print(f"Assistant: {completion.choices[0].message.content}")

        return completion.choices[0].message.content


if __name__ == "__main__":
    gpt = GPT4o(model="gpt-4o")

    #IMAGE_PATH = "/home/jcneves/datasets/Dermatology/Derm7pt/Fcl044.jpg"
    #base64_image = gpt.encode_image(image_path=IMAGE_PATH)
    #gpt.inference_vision(model="gpt-4o", base64_image=base64_image)

    instruction = f"You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    query = """###Question: What is the type of skin lesion that is associated with the presence following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""
    input_query = query.format("atypical pigment network, regular streaks, regular dots and globules, regression structures")

    response = gpt.inference_text(instruction=instruction, query=input_query, max_new_tokens=1)
    print(response)
