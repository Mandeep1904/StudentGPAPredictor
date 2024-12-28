from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

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
    return jsonify(gpa=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
