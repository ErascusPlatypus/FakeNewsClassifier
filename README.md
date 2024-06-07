# fake_news_classifier

You can visit the [website here](https://fake-news-classifier-w73t.onrender.com) 

The website is a simple implementation of the trained model. It runs on a Flask backend with the model saved and compressed to a Tflite version. Users can input the link of any news article that they want to check and the website will automatically extract the text from the website and run it through the model and output the result.

This repository contains a Jupyter Notebook with code for a Fake News Classifier that uses a Neural Network model with word embeddings to classify news articles as fake or real. 

## Dataset

The dataset is the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv) from Kaggle. It contains labeled news articles as either fake or real.

## Model

The code implements a Neural Network model that uses word vectors in the preprocessing step to transform the text data into numerical representations suitable for the model. The classifier outputs a probability indicating whether the news is fake or real.

### Performance

- **Training Accuracy:** 99.7%
- **Validation Accuracy:** 99.1%

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/fake-news-classifier.git
cd fake-news-classifier
pip install -r requirements.txt
```

## Files
- `app.py` : Flask code for the backend of the website
- `tokenizer.joblib` : tokenizer for the ML Model in a portable format
- `sentiment_model_pruned_quantized.tflite` : The ML Model saved and compressed into a smaller size
- `word_vec.ipynb` : Juoyter Notebook containing the Neural Network NLP Model
- `templates/index.html` : Front-end of the website
- `requirements.txt`: List of dependencies required to run the code.

## Acknowledgements

- The dataset is sourced from [Kaggle](https://www.kaggle.com).
- Special thanks to the creators of the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

Feel free to customize the content according to your specific needs and project structure.
