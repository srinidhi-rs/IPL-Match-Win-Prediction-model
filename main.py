### Import Required Libraries**
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

### Load and Explore the Dataset**
# Load the dataset
file_path = "ipl.csv"
df = pd.read_csv(file_path)

# Display basic dataset info
print(df.info())
print(df.head())


### Extract Match Winners**
# Extract match winners
df_winners = df.loc[df.groupby("mid")["total"].idxmax()][["mid", "batting_team"]]
df_winners = df_winners.rename(columns={"batting_team": "winner"})

# Merge winners with original data
df = df.merge(df_winners, on="mid")

###  Data Preprocessing: One-Hot Encoding for Categorical Features**

# Define categorical and numerical features
categorical_features = ["batting_team", "bowling_team", "venue"]
numerical_features = ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]

# Apply OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])

# Convert encoded features into a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Merge encoded data with numerical columns
df_encoded = pd.concat([encoded_df, df[numerical_features + ["winner"]]], axis=1)


### Splitting Data into Training and Testing Sets**

# Define feature set and target variable
features = list(encoded_df.columns)[:-1]  # All columns except 'winner'
target = "winner"

# Splitting data into training and testing sets
X = df_encoded[features]
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Training the SVM Model**
# Train an SVM model
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train, y_train)

### Evaluating the Model**

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

### Predicting the Winner for a Given Match**


def predict_winner(batting_team, bowling_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5):
    # Convert input to DataFrame
    input_data = pd.DataFrame([[batting_team, bowling_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5]],
                              columns=categorical_features + numerical_features)

    # Apply one-hot encoding
    input_encoded = encoder.transform(input_data[categorical_features])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Merge encoded categorical data with numerical features
    final_input = pd.concat([input_encoded_df, input_data[numerical_features]], axis=1)

    # Ensure column order matches training data
    final_input = final_input.reindex(columns=X_train.columns, fill_value=0)

    # Predict winner
    prediction = model.predict(final_input)
    return prediction[0]

predicted_winner = predict_winner(
    batting_team=input("Enter Batting Team: "),
    bowling_team = input("Enter Bowling Team: "),
    venue = input("Enter Venue: "),
    runs = int(input("Enter Runs Scored: ")),
    wickets = int(input("Enter Wickets Lost: ")),
    overs = float(input("Enter Overs Completed: ")),
    runs_last_5 = int(input("Enter Runs Scored in Last 5 Overs: ")),
    wickets_last_5 = int(input("Enter Wickets Lost in Last 5 Overs: ")),
    )

print(f"Predicted Winner: {predicted_winner}")
