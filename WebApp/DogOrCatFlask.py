# Import required libraries.
import os
import tensorflow as tf
from flask import request, render_template, Flask
from werkzeug.utils import secure_filename
import numpy as np
import keras
import time

# Generate timestamp.
timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

# Supress tensorflow logging messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

# Limit Tensorflow GPU memory usage.
# Allocate 3GB VRAM and set memory growth to TRUE.
GPU = tf.config.experimental.list_physical_devices("GPU")
if GPU:
    try:
        for gpu in GPU:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 3))
    except RuntimeError as e:
        print(e)

# Locate saved model.
model_file = "DogOrCat_Final.h5"
path = os.path.join("../", model_file)

# App configrations.
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xecsd]/'
app.config["UPLOAD_FOLDER"] = "static"

# Result dictionary to be sent to HTML.
results = {"PATH": 0, "PREDICTION": 0}

# Prediction function.
def predict(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150))  # Load image from path and resize.
    img_array = keras.preprocessing.image.img_to_array(img)  # Convert the images into NumPy array.
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array])  # Stack images.
    # Loading and prediction is done in the same function to avoid crash at production due to an intercommunication issue between Tensorflow and Keras.
    model = tf.keras.models.load_model(path)  # Load the model.
    classes = model.predict(images)  # Make predictions.
    if classes[0] > 0.5:
        results = {"PATH": img_path, "PREDICTION": "dog"}
    else:
        results = {"PATH": img_path, "PREDICTION": "cat"}
    return results


# Define functions to be executed at endpoints.
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            image = request.files["file"]  # Get the file.
            # Get secured file name and add timestamp to make it unique.
            filename = timestr + secure_filename(image.filename)
            image.save(os.path.join("static", filename))
            path = os.path.join("static", filename)
            result = predict(img_path=path)  # Send the image to prediction algorithm.
            return render_template("index.html", res=result)
        except:
            return render_template("index.html", res=results)

    return render_template("index.html", res=results)


# Run app.
if __name__ == "__main__":
    app.run(debug=True)
