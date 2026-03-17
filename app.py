from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Classes
classes = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot","Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight","Grape___healthy",
    "Orange___Haunglongbing","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

# Disease solutions
disease_solutions = {
    "Apple___Apple_scab": "Apply Sulfur or Captan fungicide during early growth stages.",
    "Apple___Black_rot": "Prune infected branches and apply fungicide regularly.",
    "Apple___Cedar_apple_rust": "Use resistant varieties and apply Myclobutanil fungicide.",
    "Apple___healthy": "Crop is healthy. No pesticide required.",
    "Blueberry___healthy": "Crop is healthy. Maintain proper irrigation and soil pH.",
    "Cherry___Powdery_mildew": "Apply Sulfur-based fungicide and improve air circulation.",
    "Cherry___healthy": "Crop is healthy. No treatment required.",
    "Corn___Cercospora_leaf_spot": "Use resistant hybrids and apply foliar fungicides.",
    "Corn___Common_rust": "Apply fungicide and remove infected leaves.",
    "Corn___Northern_Leaf_Blight": "Rotate crops and apply recommended fungicides.",
    "Corn___healthy": "Crop is healthy. Maintain good field hygiene.",
    "Grape___Black_rot": "Apply Mancozeb or Copper fungicide and prune infected parts.",
    "Grape___Esca_(Black_Measles)": "Remove infected vines and avoid water stress.",
    "Grape___Leaf_blight": "Apply fungicide and improve vineyard ventilation.",
    "Grape___healthy": "Crop is healthy. Maintain proper pruning schedule.",
    "Orange___Haunglongbing": "Remove infected trees immediately and control psyllid insects.",
    "Peach___Bacterial_spot": "Apply Copper-based bactericide and avoid overhead irrigation.",
    "Peach___healthy": "Crop is healthy. Maintain proper pruning and watering.",
    "Pepper,_bell___Bacterial_spot": "Apply Copper-based bactericide and use disease-free seeds.",
    "Pepper,_bell___healthy": "Crop is healthy. Maintain balanced fertilization.",
    "Potato___Early_blight": "Apply Chlorothalonil or Mancozeb fungicide.",
    "Potato___Late_blight": "Spray Metalaxyl or Mancozeb fungicide every 7 days.",
    "Potato___healthy": "Crop is healthy. No pesticide required.",
    "Raspberry___healthy": "Crop is healthy. Ensure proper spacing and sunlight.",
    "Soybean___healthy": "Crop is healthy. Maintain soil fertility and weed control.",
    "Squash___Powdery_mildew": "Apply Sulfur fungicide and ensure good air circulation.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and apply appropriate fungicide.",
    "Strawberry___healthy": "Crop is healthy. Maintain proper irrigation schedule.",
    "Tomato___Bacterial_spot": "Apply Copper-based bactericide and remove infected leaves.",
    "Tomato___Early_blight": "Use Mancozeb fungicide and remove affected foliage.",
    "Tomato___Late_blight": "Apply Chlorothalonil or Metalaxyl fungicide every 7 days.",
    "Tomato___Leaf_Mold": "Improve greenhouse ventilation and apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicide spray.",
    "Tomato___Spider_mites": "Use neem oil or insecticidal soap spray.",
    "Tomato___Target_Spot": "Apply fungicide and improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools regularly.",
    "Tomato___healthy": "Crop is healthy. No pesticide required."
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["file"]

        # 🔥 SAFE IMAGE PROCESSING
        try:
            img = Image.open(file).convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except:
            return render_template("index.html",
                                   disease=None,
                                   solution="Invalid image. Please upload a valid plant leaf image.",
                                   confidence=None)

        # 🔥 PREDICTION
        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        # 🔥 FILTER NON-LEAF / RANDOM IMAGES
        if confidence < 0.5:
            return render_template("index.html",
                                   disease=None,
                                   solution="Please upload a valid plant leaf image.",
                                   confidence=None)

        # RESULT
        disease = classes[class_index]
        solution = disease_solutions.get(disease, "No recommendation available.")

        confidence = round(confidence * 100, 2)

        return render_template("index.html",
                               disease=disease,
                               solution=solution,
                               confidence=confidence)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
