import gradio as gr
import pickle
import pandas as pd

with open("rf_model.pkl", "rb") as f:
    rf_model, training_columns = pickle.load(f)

sex_options = ["female", "male"]
smoker_options = ["no", "yes"]
region_options = ["northeast", "northwest", "southeast", "southwest"]
children_options = [0, 1, 2, 3, 4, 5]

def prepare_input(age, bmi, children, sex, smoker, region):
    data = pd.DataFrame(0, index=[0], columns=training_columns)
    if 'age' in data.columns:
        data['age'] = age
    if 'bmi' in data.columns:
        data['bmi'] = bmi
    if f"children_{children}" in data.columns:
        data[f"children_{children}"] = 1
    if f"sex_{sex}" in data.columns:
        data[f"sex_{sex}"] = 1
    if f"smoker_{smoker}" in data.columns:
        data[f"smoker_{smoker}"] = 1
    if f"region_{region}" in data.columns:
        data[f"region_{region}"] = 1
    return data

def predict_charges(age, bmi, children, sex, smoker, region):
    df = prepare_input(age, bmi, children, sex, smoker, region)
    pred = rf_model.predict(df)[0]
    return f"${pred:,.2f}" 

custom_css = """
body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; }
h1 { background: linear-gradient(90deg, #4f46e5, #6366f1); 
     color: white; padding: 20px; border-radius: 12px; text-align: center; }
.gradio-container { max-width: 800px; margin: auto; padding: 20px; }
.input_label { font-weight: bold; font-size: 14px; }
.output_label { font-size: 18px; font-weight: bold; color: #111827; }
.gr-button { background: linear-gradient(90deg,#4f46e5,#6366f1); 
             color: white; font-weight: bold; border-radius: 10px; padding: 12px; }
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("<h1>✨ Medical Insurance Charges Predictor ✨</h1>")

    with gr.Row():
        with gr.Column():
            age_input = gr.Number(label="Age", value=30)
            bmi_input = gr.Number(label="BMI", value=25)
            children_input = gr.Dropdown(children_options, label="Children", value=0)
        with gr.Column():
            sex_input = gr.Dropdown(sex_options, label="Sex", value="female")
            smoker_input = gr.Dropdown(smoker_options, label="Smoker", value="no")
            region_input = gr.Dropdown(region_options, label="Region", value="northeast")

    predict_button = gr.Button("Predict Charges")
    output = gr.Textbox(label="💰 Predicted Charges", interactive=False, elem_id="pred_output")

    predict_button.click(
        fn=predict_charges,
        inputs=[age_input, bmi_input, children_input, sex_input, smoker_input, region_input],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch()
