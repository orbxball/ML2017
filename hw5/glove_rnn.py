import sys, os
import argparse
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Activation, Dense, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K


def read_data(file):
  print('Reading training data...')
  tags, texts, categories = [], [], []
  with open(file) as f:
    f.readline()
    for line in f.readlines():
      buf = line.split('"', 2)

      tags_tmp = buf[1].split(' ')
      for category in tags_tmp:
        categories.append(category)
      tags.append(tags_tmp)
      text = buf[2][1:]
      texts.append(text)
  return tags, texts, sorted(list(set(categories)))


def read_test(file):
  print('Reading test data...')
  texts = []
  with open(file) as f:
    f.readline()
    for line in f.readlines():
      text = line.split(',', 1)[1]
      texts.append(text)
  return texts


def to_multi_categorical(tags, categories):
  categorical_tags = np.zeros((len(tags), len(categories)))
  for i, tag in enumerate(tags):
    for item in tag:
      categorical_tags[i][categories.index(item)] = 1
  return categorical_tags


def split_data(X, Y, valid_ratio):
  valid_size = int(valid_ratio * X.shape[0])
  permu = np.random.permutation(X.shape[0])
  valid_idx = permu[:valid_size]
  train_idx = permu[valid_size:]
  x_valid = X[valid_idx, :]
  y_valid = Y[valid_idx, :]
  x_train = X[train_idx, :]
  y_train = Y[train_idx, :]
  return (x_train, y_train), (x_valid, y_valid)


def get_embedding_dict(path):
  embedding_dict = {}
  with open(path, 'r') as f:
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embedding_dict[word] = coefs
  return embedding_dict


def get_embedding_matrix(word_index, embedding_dict, num_words, embedding_dim):
  embedding_matrix = np.zeros((num_words,embedding_dim))
  for word, i in word_index.items():
    if i < num_words:
      embedding_vector = embedding_dict.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
  return embedding_matrix


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f2_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return K.mean(2 * (p * r) / (p + r + K.epsilon()))

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main():
  ### read training and testing data
  train_tags, train_texts, categories = read_data(train_path)
  test_texts = read_test(test_path)
  all_corpus = train_texts + test_texts
  print ('Find {} articles.'.format(len(all_corpus)))

  ### tokenizer for all data
  if os.path.exists(tokenizer_name):
    with open(tokenizer_name, 'rb') as f:
      tokenizer = pickle.load(f)
  else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
  word_index = tokenizer.word_index

  ### convert word sequences to index sequence
  print ('Convert to index sequences.')
  train_sequences = tokenizer.texts_to_sequences(train_texts)
  test_sequences = tokenizer.texts_to_sequences(test_texts)

  ### padding to equal length
  print ('Padding sequences.')
  train_sequences = pad_sequences(train_sequences)
  max_article_length = train_sequences.shape[1]
  test_sequences = pad_sequences(test_sequences, maxlen=max_article_length)

  ### transform tags into categorical tags
  train_cato_tags = to_multi_categorical(train_tags, categories)

  ### split data into training set and validation set
  (X_train, Y_train), (X_valid, Y_valid) = split_data(train_sequences, train_cato_tags, valid_ratio)

  ### turn type
  categories = np.array(categories)

  ### build model or load model
  if os.path.exists(model_name):
    print('Loading model...')
    model = load_model(model_name, custom_objects={'f1_score': f1_score})
  else:
    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
    embedding_dict=get_embedding_dict('glove.6B.{}d.txt'.format(embedding_size))
    print ('Found {} word vectors.'.format(len(embedding_dict)))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index, embedding_dict, num_words, embedding_size)

    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])

    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=model_name,
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
    hist = model.fit(X_train, Y_train,
                     validation_data=(X_valid, Y_valid),
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])

    ################################################
    # We need to save model & categories & tokenizer
    ################################################
    model.save(model_name)
    np.save(categories_name, categories)
    with open(tokenizer_name, 'wb') as f:
      pickle.dump(tokenizer, f)

  ### predict on test data
  Y_pred = model.predict(test_sequences)
  ensure_dir(output_path)
  result = []
  for i, categorical in enumerate(Y_pred >= threshold):
    ret = []
    for category in categories[categorical]:
      ret.append(category)
    result.append('"{0}","{1}"'.format(i, " ".join(ret)))
  with open(output_path, "w+") as f:
    f.write('"id","tags"\n')
    f.write("\n".join(result))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Homework 5: Glove + RNN')
  parser.add_argument('--train', metavar='<#train data path>', type=str)
  parser.add_argument('--test', metavar='<#test data path>', type=str)
  parser.add_argument('--output', metavar='<#output path>', type=str)
  parser.add_argument('--valid', action='store_true')
  args = parser.parse_args()

  train_path = args.train
  test_path = args.test
  output_path = args.output
  is_valid = args.valid

  embedding_size = 100
  valid_ratio = 0.1
  nb_epoch = 100
  batch_size = 128
  threshold = 0.4
  base_dir = './model'
  model_name = os.path.join(base_dir, 'rnn_model.hdf5')
  categories_name = os.path.join(base_dir, 'rnn_categories.npy')
  tokenizer_name = os.path.join(base_dir, 'rnn_tokenizer')

  main()