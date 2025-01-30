import gradio as gr
import time

# Custom CSS
css = """
/* Textbox Styling */
textarea {
    background-color: #F0F8FF; /* Light blue background */
    border: 1px solid #D1E8FF; /* Light blue border */
    border-radius: 5px; /* Slightly rounded corners */
    font-size: 18px;
    padding: 10px;
}
textarea:focus {
    border-color: #80C6FF; /* Slightly darker blue on focus */
    outline: none;
}

/* Button Styling */
button.secondary {
    background-color: #007BFF; /* Professional blue */
    color: #FFFFFF; /* White text */
    border: none;
    border-radius: 5px;
    font-size: 16px;
    padding: 10px 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
}
button.secondary:hover {
    background-color: #0056B3; /* Darker blue on hover */
}
button.disabled {
    background-color: #E0E0E0; /* Light gray */
    color: #808080; /* Dark gray text */
    cursor: not-allowed;
}

#flag-correct {
    background-color: #80EF80 !important; /* Pastel green */
    flex: initial;
}

#flag-incorrect {
    background-color: #FF746C !important; /* Pastel green */
    flex: initial;
}

#component-10 {
    display: flex;
    justify-content: right;
}

/* General Notes */
ul {
    color: #333333; /* Dark gray text */
}
h1 {
    color: #000000; /* Black for maximum readability */
}

body {
    background-color: #333333;
}
"""

# Custom JS
js = """

function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}

"""

# Function to get the concept predictions from input image using ExpLICD (Step 1)
def step_1(pil_image, progress=gr.Progress()):

    predicted_concepts, raw_scores = engine.x_to_c(pil_image=pil_image, model_step_1=model_step_1)

    # Define instruction and query
    instruction = f"You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations.\n"
    query = """###Question: What is the type of skin lesion that is associated with the following dermoscopic concepts: {}. \n###Options: \nA. Nevus\nB. Melanoma. \n###Answer:"""
    
    input_query = query.format(predicted_concepts)
    final_prompt = instruction + input_query

    # Simulate processing
    progress(0, desc="Predicting clinical concepts...")
    time.sleep(1)
    progress(0.5, desc="Predicting clinical concepts...")
    time.sleep(1.5)
    progress(1, desc="Predicting clinical concepts...")

    return final_prompt

# Function to get the final diagnosis (Step 2)
def step_2(intermediate_prompt, progress=gr.Progress()):
    llm_response = engine.c_to_y(intermediate_prompt=intermediate_prompt, model_step_2=model_step_2)

    predicted_concepts = intermediate_prompt[intermediate_prompt.find("dermoscopic concepts:")+len("dermoscopic concepts:"):intermediate_prompt.find(". \n###Options:")]
    textual_explanation = predicted_concepts + f" Thus the diagnosis is {llm_response}."

    # Simulate processing
    progress(0, desc="Generating final diagnosis...")
    time.sleep(1)
    progress(0.5, desc="Generating final diagnosis...")
    time.sleep(1.5)
    progress(1, desc="Generating final diagnosis...")

    return textual_explanation

def main():
    callback = gr.CSVLogger()

    # Define the Gradio interface
    with gr.Blocks(css=css, js=js) as demo:
        # Add a title
        gr.HTML(
            """
            <h1 style="text-align: center;">Demo for Two-Step Concept-Based Skin Lesion Diagnosis</h1>
            <p style="text-align: center;">This demo showcases our two-step approach for skin lesion diagnosis. For further details, please refer to our paper.</p>

            <div id="instructions" style="margin: 20px; padding: 15px; border: 1px solid #D1E8FF; background-color: #F9F9F9; border-radius: 8px;">
                <h2 style="text-align: center; color: #007BFF;">Instructions</h2>
                <ol style="color: #333333; font-size: 16px; line-height: 1.6;">
                    <li>
                        Upload a skin lesion image from your device. Then, click the <b>"Get Concept Predictions"</b> button to process the image and generate predicted dermoscopic concepts.
                    </li>
                    <li>
                        Review the detected concepts, which will be appended to a predefined prompt. You may edit the prompt if some concepts are incorrect or require adjustment.
                    </li>
                    <li>
                        Click the <b>"Get Final Diagnosis"</b> button. The prompt will be processed by a Language Model to generate the final diagnosis based on the predicted dermoscopic concepts.
                    </li>
                    <li>
                        If you'd like to flag the final diagnosis, you can choose either <b>"Flag as Correct"</b> or <b>"Flag as Incorrect"</b>. Clicking one of these buttons will save the outputs to a CSV file for further analysis.
                    </li>
                </ol>

                <h3 style="color: #FF5733; margin-top: 20px;">Important Notes:</h3>
                <ul style="color: #333333; font-size: 14px; line-height: 1.4;">
                    <li>Please only upload images of skin lesions. Non-skin lesion images may result in inaccurate predictions.</li>
                    <li>This demo is intended for research purposes only. It should not be used for medical diagnosis or treatment.</li>
                </ul>
            </div>
            <hr>
            """
        )

        # Input image for ExpLICD
        with gr.Row():
            image_input = gr.Image(
                type="pil", 
                label="Input Image"
            )
            intermediate_output = gr.Textbox(
                label="Prompt (Editable)", 
                lines=10,
                info="This is the generated prompt for an LLM to provide the final diagnosis. It includes the predicted dermoscopic concepts, which you can modify before submitting it to the LLM."
            )
        
        # Button to process prompt through MMed
        with gr.Row():
            button_a = gr.Button("Get Concept Predictions", icon="assets/magnifying-glass-solid.svg")
            button_b = gr.Button("Get Final Diagnosis", icon="assets/check-solid.svg", interactive=False)
        
        # Output for Model B
        final_output = gr.Textbox(
            label="Final Diagnosis",
            lines=5
        )

        with gr.Row():
            btn_correct = gr.Button("Flag as Correct", elem_id="flag-correct", size="sm", icon="assets/circle-check-solid.svg")
            btn_incorrect = gr.Button("Flag as Incorrect", elem_id="flag-incorrect", size="sm", icon="assets/circle-xmark-solid.svg")

        # This needs to be called at some point prior to the first call to callback.flag()
        callback.setup([image_input, intermediate_output, final_output], "flagged_data_points")

        # State variable to manage button_b's interactivity
        model_b_enabled = gr.State(value=False)

        # Link functionality
        button_a.click(
            step_1,
            inputs=image_input,
            outputs=[intermediate_output]
        )
        button_a.click(lambda _: gr.update(interactive=True), inputs=None, outputs=button_b)
        
        button_b.click(step_2, inputs=intermediate_output, outputs=final_output)

        # We can choose which components to flag -- in this case, we'll flag all of them
        btn_correct.click(
            lambda *args: callback.flag(list(args), flag_option = ["Correct"]), 
            [image_input, intermediate_output, final_output], 
            None, 
            preprocess=False
        )

        btn_incorrect.click(
            lambda *args: callback.flag(list(args), flag_option = ["Incorrect"]), 
            [image_input, intermediate_output, final_output], 
            None, 
            preprocess=False
        )

    # Launch the app
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    import engine

    # Load models
    model_step_1, model_step_2 = engine.load_models()
    main()