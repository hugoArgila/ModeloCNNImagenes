import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image

# --- Parámetros ---
BACKGROUND_INTENSITY_THRESHOLD = 128
MIN_CONTOUR_AREA_DEFAULT = 10
CONFIDENCE_THRESHOLD_DEFAULT = 0.50

# --- Configuración de Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'modelCNNnum.h5'

# Cargar el modelo al inicio
try:
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError("No se pudo cargar la imagen.")

    avg_intensity = np.mean(img_original)
    processed_image = cv2.bitwise_not(img_original) if avg_intensity > BACKGROUND_INTENSITY_THRESHOLD else img_original

    _, thresh_initial = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)
    contours_initial, _ = cv2.findContours(thresh_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    for i_contour, contour in enumerate(contours_initial):
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA_DEFAULT:
            x, y, w, h = cv2.boundingRect(contour)
            digit_img = processed_image[y:y + h, x:x + w]

            aspect_ratio = w / h if h > 0 else 1
            if aspect_ratio > 1:
                padding = max(0, int(0.8 * h))
            else:
                padding = max(0, int(0.8 * w))

            pad_top = padding
            pad_bottom = padding
            pad_left = padding
            pad_right = padding

            digit_img_padded = cv2.copyMakeBorder(digit_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

            try:
                digit_img_resized = cv2.resize(digit_img_padded, (28, 28), interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                print(f"Error al redimensionar contorno {i_contour+1}: {e}")
                continue

            digit_img_normalized = digit_img_resized.astype('float32') / 255.0
            digit_img_reshaped = digit_img_normalized.reshape(1, 28, 28, 1)

            if model:
                prediction = model.predict(digit_img_reshaped, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence >= CONFIDENCE_THRESHOLD_DEFAULT:
                    # Convert processed contour to base64 for display
                    _, img_encoded = cv2.imencode('.png', (digit_img_resized * 255).astype(np.uint8))
                    processed_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

                    predictions.append({
                        'digit': int(predicted_digit),
                        'confidence': float(confidence),
                        'processed_image': processed_image_base64
                    })

    sorted_predictions = sorted(predictions, key=lambda item: item['processed_image']) # Sort based on processing order
    final_prediction_string = "".join(map(str, [p['digit'] for p in sorted_predictions]))

    return sorted_predictions, final_prediction_string

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    filename = None
    predictions_data = None
    final_prediction = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No file part'
            return render_template('index.html', error=error)
        file = request.files['image']
        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', error=error)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            try:
                predictions_data, final_prediction = process_image(filepath)
            except ValueError as e:
                error = str(e)
                os.remove(filepath) # Clean up the uploaded file if processing fails
                return render_template('index.html', filename=filename, error=error)
            except Exception as e:
                error = f"Error al procesar la imagen: {e}"
                os.remove(filepath) # Clean up the uploaded file if processing fails
                return render_template('index.html', filename=filename, error=error)

    return render_template('index.html', filename=filename, predictions=predictions_data, final_prediction=final_prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)