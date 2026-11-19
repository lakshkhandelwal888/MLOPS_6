
import joblib

# Load the trained model
saved_data = joblib.load("model.joblib")
model = saved_data["model"]
target_names = saved_data["target_names"]

# Accept flower measurements
print("Enter the four Iris flower measurements:")
sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

# Generate prediction
features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(features)[0]

# Display predicted species
print("Predicted class:", target_names[prediction])
