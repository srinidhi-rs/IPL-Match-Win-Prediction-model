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
df = pd.read_csv("ipl_colab.csv")  
print(df.info())  
print(df.head())  

3️⃣ Extracting Match Winners
The winner is determined based on the highest score (total column).
The dataset is updated to include the winning team for each match.
df_winners = df.loc[df.groupby("mid")["total"].idxmax()][["mid", "batting_team"]]  
df_winners = df_winners.rename(columns={"batting_team": "winner"})  
df = df.merge(df_winners, on="mid")  

4️⃣ Data Preprocessing: One-Hot Encoding for Categorical Features
Since teams and venues are categorical, One-Hot Encoding is applied to convert them into numerical format.
This allows the SVM model to process the data efficiently.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  
encoded_features = encoder.fit_transform(df[["batting_team", "bowling_team", "venue"]])  
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())  
df_encoded = pd.concat([encoded_df, df[["runs", "wickets", "overs", "runs_last_5", "wickets_last_5", "winner"]]], axis=1) 

5️⃣ Splitting Data into Training and Testing Sets
The dataset is divided into 80% training and 20% testing to ensure a fair evaluation of the model.
X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop(columns=["winner"]), df_encoded["winner"], test_size=0.2, random_state=42)  

6️⃣ Training the SVM Model
The SVC (Support Vector Classifier) with a linear kernel is used for training.
model = SVC(kernel="linear", probability=True, random_state=42)  
model.fit(X_train, y_train)

7️⃣ Evaluating the Model
The model's performance is assessed using accuracy score on the test dataset.
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy:.2f}")

8️⃣ Predicting the Winner for a Given Match
A function is created to take user input (batting team, bowling team, venue, and match progress) and predict the winner.
The function ensures that input data is preprocessed the same way as training data before prediction.
def predict_winner(batting_team, bowling_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5):  
    input_data = pd.DataFrame([[batting_team, bowling_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5]],  
                              columns=["batting_team", "bowling_team", "venue", "runs", "wickets", "overs", "runs_last_5", "wickets_last_5"])  

    input_encoded = encoder.transform(input_data[["batting_team", "bowling_team", "venue"]])  
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())  

    final_input = pd.concat([input_encoded_df, input_data[["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]]], axis=1)  
    final_input = final_input.reindex(columns=X_train.columns, fill_value=0)  

    prediction = model.predict(final_input)  
    return prediction[0]  
    
9️⃣ Interactive Prediction (User Input)
The script prompts the user to input live match details and predicts the winner in real time.
batting_team = input("Enter Batting Team: ")  
bowling_team = input("Enter Bowling Team: ")  
venue = input("Enter Venue: ")  
runs = int(input("Enter Runs Scored: "))  
wickets = int(input("Enter Wickets Lost: "))  
overs = float(input("Enter Overs Completed: "))  
runs_last_5 = int(input("Enter Runs Scored in Last 5 Overs: "))  
wickets_last_5 = int(input("Enter Wickets Lost in Last 5 Overs: "))  

predicted_winner = predict_winner(batting_team, bowling_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5)  
print(f"Predicted Winner: {predicted_winner}")  
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
