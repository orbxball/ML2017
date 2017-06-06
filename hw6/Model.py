import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import Reshape, Flatten, Lambda
from keras.layers.merge import concatenate, dot, add
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2

def build_cf_model(n_users, n_movies, dim, isBest=False):
    u_input = Input(shape=(1,))
    if isBest:
        u = Embedding(n_users, dim, embeddings_regularizer=l2(1e-5))(u_input)
    else:
        u = Embedding(n_users, dim)(u_input)
    u = Reshape((dim,))(u)
    u = Dropout(0.1)(u)

    m_input = Input(shape=(1,))
    if isBest:
        m = Embedding(n_movies, dim, embeddings_regularizer=l2(1e-5))(m_input)
    else:
        m = Embedding(n_movies, dim)(m_input)
    m = Reshape((dim,))(m)
    m = Dropout(0.1)(m)

    if isBest:
        u_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5))(u_input)
    else:
        u_bias = Embedding(n_users, 1)(u_input)
    u_bias = Reshape((1,))(u_bias)
    if isBest:
        m_bias = Embedding(n_movies, 1, embeddings_regularizer=l2(1e-5))(m_input)
    else:
        m_bias = Embedding(n_movies, 1)(m_input)
    m_bias = Reshape((1,))(m_bias)

    out = dot([u, m], -1)
    out = add([out, u_bias, m_bias])
    if isBest:
        out = Lambda(lambda x: x + K.constant(3.581712))(out)

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model


def build_deep_model(n_users, n_movies, dim, dropout=0.1):
    u_input = Input(shape=(4,))
    u = Embedding(n_users, dim)(u_input)
    # u = Reshape((dim,))(u)
    u = Flatten()(u)

    m_input = Input(shape=(19,))
    m = Embedding(n_movies, dim)(m_input)
    # m = Reshape((dim,))(m)
    m = Flatten()(m)

    out = concatenate([u, m])
    out = Dropout(dropout)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.15)(out)
    out = Dense(dim, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1, activation='relu')(out)

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model

def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

