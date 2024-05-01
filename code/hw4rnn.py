import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=1024, embed_size=300):
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
        self.recurrent_layer2 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=False)
        self.reccurrent = [self.recurrent_layer]
        ## - You may also want a dense layer near the end...
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.dense3 = tf.keras.layers.Dense(300, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.3)
        self.dense_last = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        self.fforward = [self.dense3]



    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        out = self.embed_layer(inputs)
        for recur in self.reccurrent:
            out = recur(out)
        for layer in self.fforward:
            out = layer(out)
            out = self.drop(out)
        logits = self.dense_last(out)
        return logits

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, out_file, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]
        output = ""
        i = 0

        while output.strip() != "<VER>" and i < 6:
            logits = self.call(next_input)
            logits = np.array(logits[0, 0, :])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n, p=n_logits)

            output = reverse_vocab[out_index]
            text.append(output)
            next_input = np.array([[out_index]])
            if len(text) % 9 == 0:
                out_file.write(" ".join(text))
                out_file.write("\n")
                text = []
                i += 1

        out_file.write("\n")
        out_file.write("\n")
        return " ".join(text)  # Return the generated text instead of printing


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
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss_metric,
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model=model,
        epochs=2,
        batch_size=32,
    )


#########################################################################################
def split_data(to_split, window_size):
    remainder = (len(to_split) - 1) % window_size
    to_split = to_split[:-remainder]

    X_rnn = to_split[:-1].reshape(-1, window_size)

    y_rnn = to_split[1:].reshape(-1, window_size)

    return X_rnn, y_rnn

# Define the recombine_syllables function
def recombine_syllables(syllables, dictionary):
    words = []
    current_word = ''
    for syllable in syllables:
        potential_word = current_word + syllable
        if potential_word in dictionary:
            words.append(potential_word)
            current_word = ''
        else:
            current_word = potential_word
    if current_word:
        words.append(current_word)
    return ' '.join(words)


# Add a new function to handle post-processing
def post_process_generated_text(generated_text, dictionary):
# Split the generated text into syllables
    syllables = generated_text.split()
    # Recombine the syllables into words
    recombined_text = recombine_syllables(syllables, dictionary)
    return recombined_text

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train_grammar, test_grammar, voc_grammar = get_data("../data/train_grammar.txt", "../data/test_grammar.txt")
    train, test, voc = get_data("../data/train.txt", "../data/test.txt")
    vocab_set = set(voc_grammar.keys()).union(set(voc.keys()))
    vocab_combined = {word: idx for idx, word in enumerate(vocab_set)}
    print(len(voc_grammar))
    # vocab = voc

    X0, Y0 = split_data(np.asarray(train_grammar), 20)
    X1, Y1 = split_data(np.asarray(test_grammar), 20)

    # ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab_combined)

    args.model.fit(
        X0, Y0, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X1, Y1)
    )

    args.model.save_weights('first lstm try.h5')


    for word1 in "Knight it for fear to wet a widow eye,".split():
        if word1 not in vocab_combined:
            print(f"{word1} not in vocabulary")
        else:
            # args.model.generate_sentence(word1, 20, vocab_combined, 10)
            with open("../data/model_ouput_gram.txt", "a") as f:
                generated_text = args.model.generate_sentence(word1, 20, vocab_combined, f, 10)

    X0, Y0 = split_data(np.asarray(train), 20)
    X1, Y1 = split_data(np.asarray(test), 20)

    next_input = np.array([[10]])
    args.model(next_input)

    ## TODO: Get your model that you'd like to use
    args.model.load_weights('first lstm try.h5')

    args.model.fit(
        X0, Y0, epochs=5, batch_size=args.batch_size, validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in "Is it for fear to wet a widow's eye,".split():
        if word1 not in vocab_combined:
            print(f"{word1} not in vocabulary")
        else:
            # args.model.generate_sentence(word1, 20, vocab_combined, 10)
            with open("../data/model_ouput2.txt", "a") as f:
                generated_text = args.model.generate_sentence(word1, 20, vocab_combined, f, 10)
            # if isinstance(generated_text, str):

            # # Post-process the generated text to recombine syllables
            #     dictionary = set(vocab_combined.keys())  # Use your combined vocabulary as the dictionary
            #     readable_text = post_process_generated_text(generated_text, dictionary)
            #     print(readable_text)
            # else:
            #     print("Generated text is not in the correct format.")

if __name__ == "__main__":
    main()
