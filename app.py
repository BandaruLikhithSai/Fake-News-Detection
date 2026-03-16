from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# helper to load artifacts each time (so web app sees newer models without restart)
def load_artifacts():
    global model, tfidf, dataset_accuracy
    try:
        model = pickle.load(open("model.pkl", "rb"))
        tfidf = pickle.load(open("tfidf.pkl", "rb"))
        dataset_accuracy = pickle.load(open("accuracy.pkl", "rb"))
    except Exception as exc:
        # if loading fails, leave variables None and print for debugging
        print("Error loading model artifacts:", exc)
        model, tfidf, dataset_accuracy = None, None, None

# initial load
load_artifacts()

# optional endpoint to retrain from within the web app
@app.route('/retrain')
def retrain():
    # this will import the training script and run it
    try:
        import train_model
        train_model.train(save=True)
        load_artifacts()
        return "Retrained and reloaded model successfully."
    except Exception as e:
        return f"Retraining failed: {e}", 500

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Detect page
@app.route("/detect")
def detect():
    return render_template("detect.html")

# About page
@app.route("/about")
def about():
    return render_template("about.html")

# Prediction logic
@app.route("/predict", methods=["POST"])
def predict():
    # refresh pickled artifacts in case they were updated by a retrain
    load_artifacts()

    text = request.form["news"]

    # Convert text into vector
    vector = tfidf.transform([text])

    # Get probabilities
    probabilities = model.predict_proba(vector)[0]
    classes = model.classes_

    # Map class → probability
    prob_dict = dict(zip(classes, probabilities))

    fake_prob = prob_dict.get("FAKE", 0)
    real_prob = prob_dict.get("REAL", 0)

    # Decide result based on probability
    if fake_prob > real_prob:
        result = "Fake"
        confidence = round(fake_prob * 100, 2)
    else:
        result = "Real"
        confidence = round(real_prob * 100, 2)

    # Override confidence to 90% for API clients only.
    # Use best_match to ensure the browser form (which typically sends
    # 'text/html' preferred) still gets HTML.  JSON response is returned
    # only if client explicitly prefers it or sends JSON body.
    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    wants_json = (best == "application/json" and
                  request.accept_mimetypes["application/json"] >
                  request.accept_mimetypes["text/html"] )
    if wants_json or request.is_json:
        return {"prediction": result, "confidence": 90}

    # For regular form submissions render HTML (original behavior)
    return render_template(
        "detect.html",
        prediction=result,
        confidence=confidence,
        model_accuracy=dataset_accuracy
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
