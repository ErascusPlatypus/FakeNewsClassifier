from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import joblib
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)   # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log when the server starts
logging.info('Flask Server is active')

# Load the TFLite model
tflite_model_path = 'sentiment_model_pruned_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
logging.info('TFLite model loaded successfully')

# Load the tokenizer
tokenizer = joblib.load('tokenizer.joblib')
logging.info('Tokenizer loaded successfully')

# Define the max length for padding
MAX_LEN = 250  # Replace this with your actual max length

def preprocess_text(text):
    logging.info('Preprocessing text for prediction')
    seqs = tokenizer.texts_to_sequences([text])
    padded_seqs = pad_sequences(seqs, maxlen=MAX_LEN)
    logging.info('Text preprocessing complete')
    return padded_seqs

def sentiment(review):
    # Preprocess the review text
    input_data = preprocess_text(review)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the input data
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # Run the model
    interpreter.invoke()

    # Get the prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/')
def home():
    logging.info('Home page accessed')
    return render_template('index.html')

@app.route('/extension/predict', methods=['POST'])
def extension_predict():
    logging.info('Predict route accessed')
    data = request.get_json()
    url = data.get('url')
    logging.info(f'URL received: {url}')
    
    article_text = extract_text_from_url(url)
    
    if not article_text:
        logging.warning('Could not extract text from the URL')
        return jsonify({'prediction': 'Could not extract text from the URL.'})
    
    # Make a prediction
    logging.info('Making prediction')
    prediction = sentiment(article_text)
    
    # Convert prediction to a meaningful label
    logging.info(f'Predicted value of article is : {prediction[0][0]}')
    result = 'Fake' if prediction[0][0] < 0.5 else 'Real'
    logging.info(f'Prediction complete: {result}')
    
    return jsonify({'prediction': result})


@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Predict route accessed')
    url = request.form['url']
    logging.info(f'URL received: {url}')
    
    article_text = extract_text_from_url(url)
    
    if not article_text:
        logging.warning('Could not extract text from the URL')
        return render_template('index.html', prediction='Could not extract text from the URL.')
    
    # Make a prediction
    logging.info('Making prediction')
    prediction = sentiment(article_text)
    
    # Convert prediction to a meaningful label
    logging.info(f'Predicted value of article is : {prediction[0][0]}')
    result = 'Fake' if prediction[0][0] < 0.5 else 'Real'  # Assuming binary classification with a sigmoid output
    logging.info(f'Prediction complete: {result}')
    if result == 'Fake':
        color_back = 'red_back'
        color_front = 'red_front'
    else:
        color_back = 'green_back'
        color_front = 'green_front'
    return render_template('index.html', prediction=f'This news article is {result}.', color_back=color_back, color_front=color_front)

def extract_text_from_url(url):
    try:
        logging.info(f'Extracting text from URL: {url}')
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text based on HTML structure of news websites
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        logging.info('Text extraction complete')
        return article_text
    
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)


