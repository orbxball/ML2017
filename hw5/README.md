# HW5
Determine the tags for each summary of a book.

本次作業目標是利用書籍摘要預測類型(tags)。當中有多個類別，屬於multi-label。也希望各位同學能透過此次作業熟悉機器學習在NLP上的應用。


## Kagge Link
<https://inclass.kaggle.com/c/ml2017-hw5>  

### requirement.txt
```
numpy
scipy
pandas
Keras
tensorflow
scikit-learn
h5py
Cython
nltk
```

## Best Version
<http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>  


## RNN Version
### Reading reference:
<https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py>  
<https://github.com/fchollet/keras/blob/master/tests/keras/preprocessing/text_test.py>  
<https://github.com/fchollet/keras/issues/741>  
<http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>

#### Use pre-trained word embedding in Keras
<https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html>  

#### keras metrics f1 score
<https://github.com/fchollet/keras/issues/2607>  
<https://github.com/fchollet/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py>  
<https://github.com/fchollet/keras/issues/3977>  

<https://keras.io/preprocessing/text/#tokenizer>  
<https://keras.io/preprocessing/sequence/#pad_sequences>  

<https://keras.io/layers/embeddings/#embedding>  
<https://keras.io/layers/recurrent/#lstm>  

<http://stackoverflow.com/questions/7961363/removing-duplicates-in-lists>  


## Reference
<https://www.quora.com/What-is-bagging-in-machine-learning>  
<https://www.tutorialspoint.com/python/string_split.htm>  
