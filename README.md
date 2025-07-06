# ie7500_sentiment_npl

explanation of model choice: 
BERT is a transformer-based model that has been pre-trained on a vast amount of text data. One of the primary application of BERT is sentiment analysis. Pre-trained BERT models can be fine-tuned for sentiment analysis with relatively small amounts of data. Our model, DistilBERT, is a compressed version of the BERT developed by HuggingFace. DistilBERT's reduced size makes it 60% faster while retaining 97% accuracy when compared to BERT.

training procedures:
The training of the model is in the file "DistilBERT_model_training.ipynb". The data is loaded from HuggingFace. It is tokenized with a WordPiece method which breaks words into sub-word. This aids the model in out of vocabulary words. Then the model is created and setup with our training parameters. The model is trained. The model evaluated. The F1 and accuracy was 93%.

using the model:
The file "Use_Trained_Model.ipynb" allow any two movie reviews to be inputed. Then the trained model saved in a HuggingFace repository is loaded and the two reviews are process. The model predict if the reviews are positive or negative along with its confidence.
