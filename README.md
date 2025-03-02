🏏 IPL Winning Team Prediction using SVM 🤖

📌 Project Overview

This project is an IPL match-winning team predictor built using Machine Learning (Support Vector Machines - SVM). The model is trained on historical IPL match data and uses key match statistics like runs, wickets, and venue to predict the winning team dynamically.

🚀 How It Works

This project utilizes SVM (Support Vector Machines), a supervised learning algorithm that finds the best decision boundary (hyperplane) to classify match outcomes. Given past match data, the model learns patterns and makes predictions based on input features like teams, venue, and match progress.

🔹 Why SVM for IPL Prediction?

Efficient in handling high-dimensional and complex datasets.

Finds the optimal hyperplane that separates winning and losing conditions.

Works well with categorical and numerical data after preprocessing.

🏗 Project Workflow

1️⃣ Importing Required Libraries

The project uses the following libraries:

import pandas as pd  

import numpy as np  

from sklearn.model_selection import train_test_split  

from sklearn.svm import SVC  

from sklearn.preprocessing import OneHotEncoder  

from sklearn.metrics import accuracy_score

2️⃣ Loading and Exploring the Dataset

The dataset (ipl_colab.csv) contains historical IPL match data.

It includes details like batting team, bowling team, venue, overs, and final scores.

The dataset is loaded and basic statistics are displayed. 

3️⃣ Extracting Match Winners

The winner is determined based on the highest score (total column).
The dataset is updated to include the winning team for each match.

4️⃣ Data Preprocessing: One-Hot Encoding for Categorical Features

Since teams and venues are categorical, One-Hot Encoding is applied to convert them into numerical format.
This allows the SVM model to process the data efficiently.

5️⃣ Splitting Data into Training and Testing Sets
The dataset is divided into 80% training and 20% testing to ensure a fair evaluation of the model. 

6️⃣ Training the SVM Model
The SVC (Support Vector Classifier) with a linear kernel is used for training.

7️⃣ Evaluating the Model

The model's performance is assessed using accuracy score on the test dataset.

8️⃣ Predicting the Winner for a Given Match

A function is created to take user input (batting team, bowling team, venue, and match progress) and predict the winner.
The function ensures that input data is preprocessed the same way as training data before prediction.

    
9️⃣ Interactive Prediction (User Input)

The script prompts the user to input live match details and predicts the winner in real time.

📊 Model Performance

The model demonstrates high accuracy in predicting match winners based on past IPL data. The use of One-Hot Encoding ensures efficient handling of categorical features, and SVM's ability to find the best decision boundary makes it a strong choice for classification.

⚡ Future Improvements

🔹 Use advanced models like Random Forest or Neural Networks to compare performance.

🔹 Incorporate weather conditions and player performance stats for better predictions.

🔹 Deploy as a web app or chatbot for real-time IPL predictions.

🛠 Requirements

Python 3.x

Pandas

NumPy

Scikit-learn

📌 How to Run the Project

1️⃣ Clone this repository:

2️⃣ Place your dataset (ipl_colab.csv) in the project directory.

3️⃣ Run the script:

4️⃣ Enter match details when prompted and get the predicted winner instantly!

📢 Contributing

Have ideas to improve the project? Feel free to submit a pull request or open an issue!
