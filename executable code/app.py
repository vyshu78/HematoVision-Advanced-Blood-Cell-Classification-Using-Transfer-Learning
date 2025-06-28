# app.py
from flask import Flask, render_template, request, redirect, url_for # Flask web framework components
import tensorflow as tf # To load and use the Keras model
from tensorflow.keras.preprocessing import image # For image preprocessing required by Keras model
import numpy as np # For numerical operations, especially with image arrays
import os # For interacting with the file system (saving uploads)
from PIL import Image # Pillow library for more robust image handling (e.g., converting to RGB)

app = Flask(__name__)#nitialize the Flask application

# --- Configuration for Flask App ---
MODEL_PATH = 'blood cell.h5'#Path to your saved model file
UPLOAD_FOLDER = 'static/uploads' # Directory where uploaded images will be temporarily stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} # File extensions allowed for upload
IMAGE_SIZE = (128, 128) # IMPORTANT: Must match the target_size used during model training

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Configure Flask to use the defined upload folder

# Load the trained model when the Flask app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
    # Define CLASS_NAMES in the same order as used during training
    CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    print(f"Class labels: {CLASS_NAMES}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'Blood Cell.h5' is in the same directory as app.py.")
    CLASS_NAMES = [] # Fallback if model loading fails
    model = None # Set model to None so requests gracefully handle the error

# Helper function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page (handles both GET and POST requests)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST': # This block runs when the user submits the form (uploads a file)
        # Check if 'file' part is in the request (i.e., if a file was selected)
        if 'file' not in request.files:
            return render_template('home.html', message='No file part')
        file = request.files['file'] # Get the uploaded file object

        # If user selected no file
        if file.filename == '':
            return render_template('home.html', message='No selected file')

        if file and allowed_file(file.filename):
            # Save the uploaded file temporarily to the 'static/uploads' folder
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Ensure the folder exists
            file.save(filepath) # Save the actual file

            if model is None: # Check if the model failed to load at startup
                return render_template('result.html', error_message="Model not loaded. Cannot make prediction.")

            # Preprocess the image for prediction (steps similar to training data preprocessing)
            try:
                img = Image.open(filepath).convert('RGB') # Open image using Pillow, ensure 3 channels
                img = img.resize(IMAGE_SIZE) # Resize image to the expected input size for the model
                img_array = image.img_to_array(img) # Convert Pillow image to NumPy array
                img_array = np.expand_dims(img_array, axis=0) # Add a batch dimension (model expects batch of images)
                img_array /= 255.0 # Normalize pixel values to [0, 1]
            except Exception as e:
                print(f"Error processing image: {e}")
                return render_template('home.html', message=f'Error processing image: {e}')

            # Make prediction using the loaded model
            predictions = model.predict(img_array) # Model predicts probabilities for each class
            predicted_class_index = np.argmax(predictions[0]) # Get the index of the class with highest probability
            predicted_class_label = CLASS_NAMES[predicted_class_index] # Map index to human-readable label
            confidence = predictions[0][predicted_class_index] * 100 # Calculate confidence percentage

            # Render the result page with the prediction and image path
            return render_template('result.html',
                                   image_path=url_for('static', filename=f'uploads/{filename}'),
                                   prediction=predicted_class_label,
                                   confidence=f"{confidence:.2f}%")
        else:
            # If file extension is not allowed
            return render_template('home.html', message='Allowed image types are png, jpg, jpeg')

    return render_template('home.html') # This block runs for GET requests (initial page load)

if __name__== '__main__':
    # Ensure the upload folder exists when the app starts
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) # Run the Flask development server. debug=True enables auto-reloading and debug info.