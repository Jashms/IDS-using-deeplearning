from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("cnn_lstm_intrusion_detection.keras")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON request
        data = request.get_json()
        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # Convert input to numpy array
        input_data = np.array([features])

        # Make a prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return the result
        return jsonify({
            "prediction": int(predicted_class),
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)