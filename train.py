import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("accidents_india.csv")

# Remove missing values
df.dropna(inplace=True)

# Encode categorical columns
day_encoder = LabelEncoder()
df["Day"] = day_encoder.fit_transform(df["Day_of_Week"])
df.drop("Day_of_Week", axis=1, inplace=True)

light_encoder = LabelEncoder()
df["Light"] = light_encoder.fit_transform(df["Light_Conditions"])
df.drop("Light_Conditions", axis=1, inplace=True)

severity_encoder = LabelEncoder()
df["Severity"] = severity_encoder.fit_transform(df["Accident_Severity"])
df.drop("Accident_Severity", axis=1, inplace=True)

# Features and Target
X = df.drop(
    ["Pedestrian_Crossing", "Special_Conditions_at_Site", "Severity"],
    axis=1,
)
y = df["Severity"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train model
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model
with open("test1.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ test1.pkl created successfully!")

# Test loading model
with open("test1.pkl", "rb") as f:
    loaded_model = pickle.load(f)

sample = [[2, 10, 201, 10, 10, 8, 3]]

prediction = loaded_model.predict(sample)
print("Sample Prediction:", prediction)