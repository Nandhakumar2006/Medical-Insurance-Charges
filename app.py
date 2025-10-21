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

    child_col = f"children_{children}"
    if child_col in data.columns:
        data[child_col] = 1

    sex_col = f"sex_{sex}"
    if sex_col in data.columns:
        data[sex_col] = 1

    smoker_col = f"smoker_{smoker}"
    if smoker_col in data.columns:
        data[smoker_col] = 1

    region_col = f"region_{region}"
    if region_col in data.columns:
        data[region_col] = 1

    return data

def predict_charges(age, bmi, children, sex, smoker, region):
    df = prepare_input(age, bmi, children, sex, smoker, region)
    pred = rf_model.predict(df)[0]
    return round(pred, 2)

iface = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="BMI"),
        gr.Dropdown(children_options, label="Children"),
        gr.Dropdown(sex_options, label="Sex"),
        gr.Dropdown(smoker_options, label="Smoker"),
        gr.Dropdown(region_options, label="Region")
    ],
    outputs=gr.Number(label="Predicted Charges"),
    title="Medical Insurance Charges Predictor",
    description="Predict insurance charges using a trained Random Forest model with one-hot encoded features."
)

if __name__ == "__main__":
    iface.launch()
