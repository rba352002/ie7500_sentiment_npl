# IMDB Movie Review Sentiment Analysis
This project applies different machine learning and deep learning techniques to sentiment analysis on the IMDB movie review dataset. We implemented:

Naive Bayes – A classic probabilistic classifier for text classification.

LSTM (Long Short-Term Memory) – A recurrent neural network designed for sequential data and capturing long-term dependencies.

DistilBERT – A transformer-based model, a smaller and faster version of BERT, for state-of-the-art NLP performance.

We compare these models on accuracy and F1 score to determine the most effective approach.


explanation of model choice: 
BERT is a transformer-based model that has been pre-trained on a vast amount of text data. One of the primary application of BERT is sentiment analysis. Pre-trained BERT models can be fine-tuned for sentiment analysis with relatively small amounts of data. Our model, DistilBERT, is a compressed version of the BERT developed by HuggingFace. DistilBERT's reduced size makes it 60% faster while retaining 97% accuracy when compared to BERT.

training procedures:
The training of the model is in the file "DistilBERT_model_training.ipynb". The data is loaded from HuggingFace. It is tokenized with a WordPiece method which breaks words into sub-word. This aids the model in out of vocabulary words. Then the model is created and setup with our training parameters. The model is trained. The model evaluated. The F1 and accuracy was 93%.

using the model:
The file "Use_Trained_Model.ipynb" allow any two movie reviews to be inputed. Then the trained model saved in a HuggingFace repository is loaded and the two reviews are process. The model predict if the reviews are positive or negative along with its confidence.

Training: Tokenized using WordPiece, trained with our parameters, and evaluated with an F1 score of 97%.

Files:

Training: DistilBERT_model_training.ipynb

Usage: Use_Trained_Model.ipynb

________________________________________________________________________________________
explanation of model choice (LSTM):
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) specifically designed to handle sequential data and long-term dependencies, making them highly effective for natural language processing tasks like sentiment analysis. LSTMs are particularly suited for cases where word order and contextual memory matter, such as movie reviews. We chose LSTM because it can learn temporal dynamics from sequences of word embeddings and provides strong performance on text classification tasks when trained on even moderately sized datasets. Although LSTMs do not match the speed or scale of transformer-based models like BERT, they offer a simpler and interpretable architecture that is well-suited for our 3,000-review training sample.

training procedures:
The training of the model is completed in a Google Colab notebook. First, the IMDB dataset was uploaded manually and loaded using pandas. The sentiment column was mapped to binary labels (1 for positive, 0 for negative), and the dataset was split into a training set of 3,000 samples and a test set of the remaining 47,000 samples.

We used Keras’ Tokenizer to convert the reviews into sequences of integers, with a vocabulary size of 10,000. Each review sequence was padded to a uniform length of 200 tokens. The model architecture consisted of an Embedding layer (64-dimensional vectors for each word), a single LSTM layer with 64 units, a Dropout layer (0.5) to reduce overfitting, and a final Dense layer with a sigmoid activation for binary classification. The model was compiled using the binary_crossentropy loss function and the adam optimizer, and trained with an 80/20 validation split and early stopping enabled to prevent overfitting. The training was run for up to 10 epochs with a batch size of 64.

using the LSTM model:
The trained LSTM model was evaluated on the test set of reviews. Predictions were generated, and binary labels were assigned based on a 0.5 probability threshold. The model’s performance was evaluated using the F1 score to balance precision and recall. The final F1 score achieved was [insert actual F1 score here once training finishes], reflecting the model’s ability to effectively classify sentiment in movie reviews.

The model can be further improved by experimenting with bidirectional LSTM layers, increasing the embedding dimensions, or using pre-trained word embeddings like GloVe.

Training:

Tokenized reviews with Keras tokenizer (vocab size = 10,000)

Single LSTM layer (64 units), dropout layer (0.5), final dense sigmoid layer

Trained with binary crossentropy loss, Adam optimizer, 80/20 train-test split

Files: IE7500_Project_LSTM_Model_Testing.ipynb
_________________________________________________________________________________________

Contributors
Min Chang – LSTM model implementation, evaluation

Sharmishta Kumaresh – Naive Bayes model, data preprocessing

Richard Anderson – DistilBERT model, results analysis








