import numpy as np
from keras.layers import Input, Embedding, Reshape, Dense, Dropout
from keras.layers.merge import concatenate, dot, add
from keras.models import Model

def build_cf_model(n_users, n_movies, dim):
    u_input = Input(shape=(1,))
    u = Embedding(n_users, dim)(u_input)
    u = Reshape((dim,))(u)

    m_input = Input(shape=(1,))
    m = Embedding(n_movies, dim)(m_input)
    m = Reshape((dim,))(m)

    # u_bias = Embedding(n_users, 1)(u_input)
    # u_bias = Reshape((1,))(u_bias)
    # m_bias = Embedding(n_movies, 1)(m_input)
    # m_bias = Reshape((1,))(m_bias)

    out = dot([u, m], -1)
    # out = add([out, u_bias, m_bias])

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model


def build_deep_model(n_users, n_movies, dim, dropout=0.1):
    u_input = Input(shape=(1,))
    u = Embedding(n_users, dim)(u_input)
    u = Reshape((dim,))(u)

    m_input = Input(shape=(1,))
    m = Embedding(n_movies, dim)(m_input)
    m = Reshape((dim,))(m)

    out = concatenate([u, m])
    out = Dropout(dropout)(out)
    out = Dense(dim, activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(1, activation='relu')(out)

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model

def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

