
import gradio as gr
import pandas as pd
import pickle
def  loaded_object(filepath='Gradio_toolkit'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object
loaded_object =  loaded_object()
scaler =loaded_object["scaler"]
model = loaded_object["model"]
encode  = loaded_object["encoder"]

inputs= [ 'gender', 'SeniorCitizen', 'Partner', 'onehotencoder__Dependents_Yes',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'onehotencoder__Contract_Oneyear', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

categoricals = ['gender','Partner','Dependents','PhoneService', 'MultipleLines','InternetService',
              'OnlineSecurity','OnlineBackup','TechSupport','StreamingTV','StreamingMovies',
            'Contract','PaperlessBilling','PaymentMethod']

"Cleaning, Processing and Feature Engineering of the input dataset."
""":dataset pandas.DataFrame"""
def predict(*args,scaler = scaler, model =model, encoder = encode):
    # Creating a dataframe of inputs
    input_data = pd.DataFrame([args], columns=inputs)
    print(input_data)
    input_data= scaler.transform(input_data)
    input_data= encoder.transform(input_data[categoricals])
    input_data = pd.DataFrame(input_data, columns=encoder.get_feature_names_out().tolist())
    input_data= input_data.join(input_data)
    input_data.drop(columns=categoricals, inplace=True)
    # Modeling
    model_output = model.predict(input_data)
    model_output = model.predict(X)
    output_str = "Hey there.....👋 your customer will"
    return(output_str,model_output)

yes_or_no = ["Yes", "No"] # To be used for the variables whose possible options are "Yes" or "No".
internet_service_choices = ["Yes", "No", "No internet service"] # A variable for the choices available for the "Internet Service" variable

# ----- App Interface
# Inputs
gender = gr.Dropdown(label="Gender", choices=["Female", "Male"], value="Female") # Whether the customer is a male or a female
SeniorCitizen = gr.Radio(label="Senior Citizen", choices=yes_or_no, value="No") # Whether a customer is a senior citizen or not
Partner = gr.Radio(label="Partner", choices=yes_or_no, value="No") # Whether the customer has a partner or not
onehotencoder__Dependents_Yes = gr.Radio(label="Dependents", choices=yes_or_no, value="Yes") # Whether the customer has dependents or not

tenure = gr.Slider(label="Tenure (months)", minimum=1, step=1, interactive=True, value=1, maximum= 72) # Number of months the customer has stayed with the company

PhoneService = gr.Radio(label="Phone Service", choices=yes_or_no, value="Yes") # Whether the customer has a phone service or not
MultipleLines = gr.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"], value="No") # Whether the customer has multiple lines or not

InternetService = gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"], value="Fiber optic") # Customer's internet service provider
OnlineSecurity = gr.Dropdown(label="Online Security", choices=internet_service_choices, value="No") # Whether the customer has online security or not
OnlineBackup = gr.Dropdown(label="Online Backup", choices=internet_service_choices, value="No") # Whether the customer has online backup or not
TechSupport = gr.Dropdown(label="Tech Support", choices=internet_service_choices, value="No") # Whether the customer has tech support or not
StreamingTV = gr.Dropdown(label="TV Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming TV or not
StreamingMovies = gr.Dropdown(label="Movie Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming movies or not

onehotencoder__Contract_Oneyear = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"], value="Month-to-month", interactive= True) # The contract term of the customer
PaperlessBilling = gr.Radio(label="Paperless Billing", choices=yes_or_no, value="Yes") # Whether the customer has paperless billing or not
PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check") # The customer's payment method
MonthlyCharges = gr.Slider(label="Monthly Charges", step=0.05, maximum= 7000) # The amount charged to the customer monthly
TotalCharges = gr.Slider(label="Total Charges", step=0.05, maximum= 10000) # The total amount charged to the customer


# Output
gr.Interface(inputs=[gender, SeniorCitizen, Partner, onehotencoder__Dependents_Yes, tenure, PhoneService,
                     MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                     TechSupport, StreamingTV, StreamingMovies,onehotencoder__Contract_Oneyear	, PaperlessBilling, PaymentMethod,
                     MonthlyCharges, TotalCharges],
             outputs = gr.Label("Awaiting Submission..."),
            fn=predict, 
            title= "Telecom Customer Churn Prediction App", 
            description= """This app uses a machine learning model to predict whether or not a customer will churn based on inputs made by you, the user. The (XGBoost) model was trained and built based on the Telecom Churn Dataset"""
            ).launch(share = True, debug =True,server_port= 5060)






