# **Text Classification Model for Gender Prediction**

This repository contains code for a text classification model for Italian texts using deep learning techniques. The model is designed to classify texts based on gender (male or female). The model was trained on a dataset containing Italian texts from various sources such as social media posts, blogs, and news articles.

## **Requirements**

- Python 3.x
- NumPy
- TensorFlow 2.x
- Keras
- Pandas
- Scikit-learn
- Gensim
- NLTK
- Matplotlib


## **Model Architecture**

The text classification model uses a combination of word embeddings and convolutional neural networks (CNNs). The model was trained on a Word2Vec embedding model which was pre-trained on a large Italian corpus. The CNN layers were used to extract features from the embeddings, which were then passed through a dense layer to produce the final output.

## **Running the Code**

Run the **`ProgettoGenderPrediction.py`** file. The code will automatically load the dataset, preprocess the texts, and train the model. After training, the model will be evaluated on a test set, and the classification report and accuracy score will be printed to the console.

Note that the model may take a while to train depending on the size of the dataset and the complexity of the model architecture.
