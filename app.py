from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Function to classify GPA
def classify_gpa(gpa):
    if gpa >= 3.5:
        return "High"
    elif gpa >= 2.5:
        return "Moderate"
    else:
        return "Low"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    inputs = [
        float(request.form["study_hours"]),
        float(request.form["extracurricular_hours"]),
        float(request.form["sleep_hours"]),
        float(request.form["social_hours"]),
        float(request.form["physical_activity"]),
    ]
    
    # Predict GPA
    prediction = model.predict([inputs])[0]
    
    # Classify the GPA
    gpa_class = classify_gpa(prediction)

    return jsonify(gpa=round(prediction, 2), gpa_class=gpa_class)

if __name__ == "__main__":
    app.run(debug=True)
