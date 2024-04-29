import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        self.embed_layer = tf.keras.layers.Embedding(
            self.vocab_size, self.embed_size, embeddings_initializer="uniform"
        )
        ## (x, y) -> ? -> (y, vocab_size) -> softmax
        ## - Define a recurrent component to reason with the sequence of data.
        #self.recurrent_layer = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=False)
        self.recurrent_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=False)
        ## - You may also want a dense layer near the end...
        self.dense1 = tf.keras.layers.Dense(64, activation="leaky_relu")
        # self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.dense3 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        embedded = self.embed_layer(inputs)
        rec_out1 = self.recurrent_layer(embedded)
        dout1 = self.dense1(rec_out1)
        logits = self.dense3(dout1)
        return logits

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0, 0, :])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n, p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]])

        print(" ".join(text))


#########################################################################################

def perplexity(y_true, y_pred):
    sparse_cross_entr = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )
    perplex = tf.exp(tf.reduce_mean(sparse_cross_entr))  # what axis?

    return perplex

def get_text_model(vocab):
    """
    Tell our autograder how to train and test your model!
    """

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    acc_metric = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.025),
        loss=loss_metric,
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model=model,
        epochs=1,
        batch_size=100,
    )


#########################################################################################
def split_data(to_split, window_size):
    remainder = (len(to_split) - 1) % window_size
    to_split = to_split[:-remainder]

    X_rnn = to_split[:-1].reshape(-1, window_size)

    y_rnn = to_split[1:].reshape(-1, window_size)

    return X_rnn, y_rnn

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train, test, voc = get_data("../data/train.txt", "../data/test.txt")
    vocab = voc

    X0, Y0 = split_data(np.asarray(train), 20)
    X1, Y1 = split_data(np.asarray(test), 20)

    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in "speak to this brown deep learning student".split():
        if word1 not in vocab:
            print(f"{word1} not in vocabulary")
        else:
            args.model.generate_sentence(word1, 20, vocab, 10)


if __name__ == "__main__":
    main()
