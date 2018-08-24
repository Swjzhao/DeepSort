import tensorflow as tf
import numpy as np
from random import randint
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, Reshape,TimeDistributed


def one_hot_encode(X, len, num):
    x = np.zeros((batch_size, len, num), dtype=np.float32)
    for i, batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[i, j, elem] = 1
    return x


def batch_gen(batch_size, len, num):
    x = np.zeros((batch_size, len, num), dtype=np.float32)
    y = np.zeros((batch_size, len, num), dtype=np.float32)
    X = np.random.randint(num, size=(batch_size, len))
    Y = np.sort(X, axis=1)

    x = one_hot_encode(X, len, num)
    y = one_hot_encode(Y, len, num)

    return x, y


def create_model(len, num):
    # attempt to create seq2seq using tensorflow and bidirectional rnn
    # learned from https://github.com/llSourcell/seq2seq_model_live/blob/master/2-seq2seq-advanced.ipynb

    #  encoder_hidden_units = num
    #  decoder_hidden_units = encoder_hidden_units*2
    #
    #  encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    #  encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
    #  decoder_inputs= tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inoputs')
    #
    #
    # # decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    #
    #  embeddings = tf.Variable(tf.random_uniform([num, len], 0, 9), dtype=tf.float32)
    #  #embeddings = tf.Variable(tf.random_uniform([num, len], 0, 9), dtype=tf.float32)
    #  encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    #
    #  decoder_inputs = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    #
    #
    #  from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
    #
    #  encoder_cell = LSTMCell(encoder_hidden_units)
    #  ((encoder_fw_outputs,
    #    encoder_bw_outputs),
    #   (encoder_fw_final_state,
    #    encoder_bw_final_state)) = (
    #      tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
    #                                      cell_bw=encoder_cell,
    #                                      inputs=encoder_inputs_embedded,
    #                                      sequence_length=encoder_inputs_length,
    #                                      dtype=tf.float32, time_major=True)
    #  )
    #
    #  encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),2)
    #  encoder_final_state_c = tf.concat(
    #      (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
    #
    #  encoder_final_state_h = tf.concat(
    #      (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    #  encoder_final_state = LSTMStateTuple(
    #      c=encoder_final_state_c,
    #      h=encoder_final_state_h
    #  )
    # encoder_states = [encoder_final_state_h, encoder_final_state_c]
    encoder_inputs = Input(shape=(None, len))
    decoder_inputs = Input(shape=(None, len))

    encoder_cell = LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder_cell(encoder_inputs)
    encoder_states = [state_h, state_c]



    encoder_cell = LSTM(128,return_sequences=True)(encoder_inputs)

    # encoder_cell = Reshape((180,))(encoder_cell )
    decoder_cell = RepeatVector(len)(encoder_outputs)

    # decoder_inputs = Input(shape=(None, len))
    decoder_cell = LSTM(128, return_sequences=True)(decoder_cell, initial_state=encoder_states)

   # decoder_outputs, _, _ = decoder_cell(decoder_inputs, initial_state=encoder_states)
    dense = TimeDistributed(layer=Dense(num, activation='softmax'))
    #dense = Dense(num, activation='softmax')
    logits = dense(decoder_cell)

    #   encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

    #    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], 0, 9), dtype=tf.float32)
    #    b = tf.Variable(tf.zeros([num]), dtype=tf.float32)
    return Model(encoder_inputs, logits)


if __name__ == "__main__":
    batch_size = 32
    len = 100  # (length of array)
    num = 100  # (max length of number)
    model = create_model(len, num)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    for iter in range(100000):
        X, Y = batch_gen(batch_size, len, num)

        loss, acc = model.train_on_batch(X, Y)

        if iter % 100 == 0:
            r = randint(2, len)
            testX = np.random.randint(num, size=(1, r))
            #print(testX)
            for i in range(len - r):
                testX = np.append(testX, [0])

            testX = testX.reshape((1, len))
            print(testX)
            test = one_hot_encode(testX, len, num)  # one-hot encoded version
            loss = '{:4.3f}'.format(loss)
            acc = '{:4.3f}'.format(acc)
            print("Iteration:", iter, "/100000  loss:", loss, "  accuracy:", acc)
            y = model.predict(test, batch_size=1)
            np_sorted = np.sort(testX)[0]
            rnn_sorted = np.argmax(y, axis=2)[0]

            for i in range(len):
                if np_sorted[0] == 0:
                    np_sorted = np.delete(np_sorted, 0)
            for i in range(len):
                if rnn_sorted[0] == 0:
                    rnn_sorted = np.delete(rnn_sorted, 0)
            is_equal = np.array_equal(np_sorted, rnn_sorted)

    model.save("deepsort.h5")
