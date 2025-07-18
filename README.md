# Crop Recommendation System

This project develops a machine learning model to recommend the most suitable crop to grow based on various environmental parameters.

# Problem

Choosing the right crop for a given location is crucial for successful farming. This project aims to build a system that can recommend crops based on soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall.


# Data
The dataset used in this project is "Crop_recommendation.csv".

It contains the following columns:

N: Nitrogen content in the soil

P: Phosphorus content in the soil

K: Potassium content in the soil

temperature: Temperature in Celsius

humidity: Relative humidity in %

ph: pH value of the soil

rainfall: Rainfall in mm

label: The recommended crop (target variable)

# Methods

Data Loading and Exploration: The dataset is loaded using pandas and basic exploration is performed.

Encoding: The categorical target variable 'label' is encoded using LabelEncoder.

Scaling: Numerical features are scaled using MinMaxScaler to bring them to a similar range.

Train-Test Split: The data is split into training and testing sets.

Model Training: Several classification models are trained and evaluated:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Model Evaluation: The performance of each model is evaluated using classification reports and confusion matrices.

Model Selection: The Gradient Boosting Classifier is selected as the best performing model.

Saving Artifacts: The trained model, scaler, and encoder are saved using pickle for later use in inference.

# Results

The Gradient Boosting Classifier achieved high accuracy on the test set, demonstrating its effectiveness in recommending crops based on the provided environmental parameters.

# Inference

The trained model can be used to predict the recommended crop for new input data. A function predict_crop is provided to take the environmental parameters as input, preprocess them using the saved scaler and encoder, and return the predicted crop.

# How to Use

Clone the repository.

Install the required libraries (pandas, scikit-learn, numpy).

Ensure you have the Crop_recommendation.csv file in the project directory.

Run the Jupyter notebook or Python script to train the model and save the artifacts.

Use the predict_crop function with your desired input values to get crop recommendations.

# Files

Crop_recommendation.csv: The dataset.

crop_recommendation_notebook.ipynb: The Jupyter notebook containing the code for the project.

models/encoder.pkl: Saved LabelEncoder object.

models/model_gbc.pkl: Saved Gradient Boosting Classifier model.

models/scaler.pkl: Saved MinMaxScaler object.

# Dependencies

pandas

numpy

scikit-learn

# Conclusion

This project successfully developed and implemented a crop recommendation system using a Gradient Boosting Classifier. The model demonstrates high accuracy in predicting the most suitable crop based on environmental conditions. This system can potentially assist farmers in making informed decisions to optimize crop yields and resource utilization
