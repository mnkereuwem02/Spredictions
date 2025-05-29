Sales Prediction App

The Sales Prediction App is a web-based application that estimates future sales based on user-provided input data. It utilizes a trained machine learning model (Extra Tree Regressor) to generate precise and reliable sales forecasts.
Project Structure

Here’s an overview of the key components:

-app.py: The main application script that runs the sales prediction app using Streamlit.
- model.ipynb: A Jupyter Notebook detailing the data analysis, model training, testing, and evaluation.
- encoders.pkl: A serialized file containing encoders for categorical features, used to prepare input data for prediction.
- scaler.pkl: A serialized scaler used to normalize numerical features for consistent and accurate model predictions.
- requirements.txt: Lists all Python dependencies required to run the application.
- .gitignore: Specifies files and directories (e.g., virtual environments) to be ignored by Git.
├── .gitignore  
├── app.py  
├── model.ipynb  
├── encoders.pkl  
├── scaler.pkl  
└── requirements.txt  

Setup and Installation
------------------------
Prerequisites:
- Python 3.7 or later
- pip (Python package installer)
- Virtualenv (recommended)


Usage

Running the Sales Prediction Application:
1. The main application is contained in app.py.
2. To run the application, execute:
   python app.py
   
   The application will:
   - Load the pre-trained encoders (encoders.pkl) and scaler (scaler.pkl).
   - Read and preprocess the input sales data (ensure your data is formatted as expected).
   - Predict and output sales forecasts.

Exploring and Modifying the Model:
1. Open the model.ipynb Jupyter Notebook.
2. Run through the notebook to review the steps for data preprocessing, model training, and evaluation.
3. Feel free to modify the code cells to experiment with different features or models.


# Deployment Disclaimer#

Initially, the project requirements specified that the web application should be deployed on AWS. While I successfully built and deployed the Sales Prediction App on an AWS EC2 instance, recurring stability issues (such as frequent shutdowns and service interruptions) impacted the user experience.

To ensure a more reliable, accessible, and seamless experience, I chose to redeploy the app on Streamlit Cloud, a free, robust hosting platform for web applications. This switch allows the app to meet all project goals while maintaining smooth functionality in data processing and sales predictions.

All core features, including the Streamlit-based UI and predictive analytics, remain fully aligned with the original project objectives.

You can view the deployed application here:  
