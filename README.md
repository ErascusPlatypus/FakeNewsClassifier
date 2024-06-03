# fake_news_classifier

This repository contains the code for a Fake News Classifier that uses a Neural Network model to classify news articles as fake or real. The dataset used for training and testing the model is sourced from [Kaggle](https://www.kaggle.com). 

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

## Usage

1. **Preprocess the Data:** Run the script to preprocess the dataset and prepare it for training.

```sh
python preprocess_data.py
```

2. **Train the Model:** Train the Neural Network model on the preprocessed data.

```sh
python train_model.py
```

3. **Evaluate the Model:** Evaluate the model's performance on the validation set.

```sh
python evaluate_model.py
```

## Files

- `preprocess_data.py`: Script to preprocess the dataset.
- `train_model.py`: Script to train the Neural Network model.
- `evaluate_model.py`: Script to evaluate the model's performance.
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
