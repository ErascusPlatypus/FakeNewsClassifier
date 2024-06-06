from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import logging
import combine_files

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log when the server starts
logging.info('Flask Server is active')

# Load the Keras model
# model = load_model('fake_news_classifier.h5')
# model = load_model('news_classifier')

# URL of the shared file on Google Drive
# file_url = "https://drive.google.com/uc?id=15WnxStJuD7f8hErFbeurC4lQXxpbQqVc"

# # Download the file
# r = requests.get(file_url, stream=True)

# # Save the file locally
# with open("classifier.h5", "wb") as f:
#     f.write(r.content)

# model = load_model('classifier.h5')

# Google Drive file ID
# file_id = "15WnxStJuD7f8hErFbeurC4lQXxpbQqVc"
# file_url = f"https://drive.google.com/uc?id={file_id}"
# output = "classifier.h5"

# # Download the file
# if not os.path.exists(output):
#     gdown.download(file_url, output, quiet=False)

# # Load the Keras model
# model = load_model(output)

combine_files.join_files()

model = load_model("fake_news_classifier - Copy_2.h5")
logging.info('Model loaded successfully')

# Load the tokenizer
tokenizer = joblib.load('tokenizer.joblib')
logging.info('Tokenizer loaded successfully')

# Define the max length for padding
MAX_LEN = 250  # Replace this with your actual max length

def sentiment(reviews):
    logging.info('Preprocessing text for prediction')
    seqs = tokenizer.texts_to_sequences(reviews)
    seqs = pad_sequences(seqs, maxlen=MAX_LEN)
    logging.info('Text preprocessing complete')
    return model.predict(seqs)

@app.route('/')
def home():
    logging.info('Home page accessed')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Predict route accessed')
    url = request.form['url']
    logging.info(f'URL received: {url}')
    
    article_text = extract_text_from_url(url)
    
    if not article_text:
        logging.warning('Could not extract text from the URL')
        return render_template('index.html', prediction='Could not extract text from the URL.')
    
    # Preprocess the text and make a prediction
    logging.info('Making prediction')
    prediction = sentiment([article_text])
    
    # Convert prediction to a meaningful label
    logging.info(f'Predicted value of article is : {prediction[0][0]}')
    result = 'Fake' if prediction[0][0] < 0.5 else 'Real'  # Assuming binary classification with a sigmoid output
    logging.info(f'Prediction complete: {result}')
    if result == 'Fake' :
        color_back = 'red_back'
        color_front = 'red_front'
    else :
        color_back = 'green_back'
        color_front = 'green_front'
    return render_template('index.html', prediction=f'This news article is {result}.', color_back=color_back, color_front=color_front )

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
    # combine_files.join_files()
    app.run(debug=True)
