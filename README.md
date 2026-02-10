\# Disease Prediction System



This project is a Machine Learning based Disease Prediction System that predicts the most likely disease based on symptoms entered by the user. The system uses a trained Random Forest model and provides fast and accurate predictions.



\## Problem Statement

Early symptoms of diseases are often ignored or misunderstood. This project aims to predict possible diseases using machine learning so that users can get early awareness and basic guidance.



\## Features

\- Predicts disease based on multiple symptoms

\- Uses Random Forest Classifier for high accuracy

\- Encodes symptoms using MultiLabelBinarizer

\- Simple and interactive Streamlit web application

\- Trained model saved using Joblib



\## Tech Stack

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Streamlit



\## Project Structure

\- app.py : Streamlit web application

\- train\_model.py : Model training script

\- Dataset.csv : Disease and symptoms dataset

\- Disease\_model.joblib : Trained ML model

\- Disease\_encoder.joblib : Disease label encoder

\- Symptom\_encoder.joblib : Symptom encoder

\- predict\_db : Stores prediction related data

\- README.md : Project documentation



\## Machine Learning Workflow

1\. Load and clean dataset

2\. Encode symptoms using MultiLabelBinarizer

3\. Encode disease labels

4\. Split data into training and testing sets

5\. Train Random Forest Classifier

6\. Evaluate model using accuracy and confusion matrix

7\. Save trained model and encoders



\## How to Run the Project

1\. Install required libraries:

&nbsp;  pip install pandas numpy scikit-learn streamlit joblib matplotlib



2\. Run the Streamlit app:

&nbsp;  streamlit run app.py



3\. Enter symptoms and get disease prediction.



\## Output

The application predicts the most probable disease based on the symptoms selected by the user.



\## Use Cases

\- Early disease awareness

\- Educational purpose for ML students

\- Healthcare-related ML applications



\## Future Improvements

\- Add more diseases and symptoms

\- Improve accuracy using deep learning models

\- Deploy application online

\- Add medical suggestions and precautions



\## Author

Rohit Garg  

B.Tech CSE Student



