import re
import pyphen
import os
import nltk
from hyphenate import hyphenate_word

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('wordnet')



# file_path = "../utils/Shakespeare/Poems/A LOVER'S COMPLAINT.txt"
def preprocess_text(file_path, is_poem=True):
    dic = pyphen.Pyphen(lang="en")
    with open(file_path, "r", encoding="utf-8") as file:
        stemmer = PorterStemmer()
        text = file.read()
        text = text.lower()
        text = text.replace("'''", "'")
        text = text.replace("\n", " <br> ") + " <STOP>"
        text = re.sub(r"\d+", "<NUM>", text)
        text = re.sub(r"\b[A-Z][a-z]*\b", "<UNK>", text)
        # text = " <SEP> ".join(text.split())
        # words = text.split()
        # # Hyphenate
        # text = " ".join([dic.inserted(word, hyphen=" ") for word in words])
        words = word_tokenize(text)

# Stem each word in the list of tokenized words
        stemmed_words = [stemmer.stem(word) for word in words]

# Rejoin the stemmed words
       
        if is_poem:
            hyphenate_words = []
            for word in stemmed_words:
                hyphenated = hyphenate_word(word)
                hyphenate_words += hyphenated
            stemmed_words = hyphenate_words

        text = ' '.join(stemmed_words)
        text = text.replace("< br >", "\n")
        text = text.replace("< num >", "<GO>" )

        return text


# preprocessed_text = preprocess_text(file_path)

# with open('poems_verses.txt', 'w', encoding='utf-8') as file:
#    file.write(preprocessed_text)


def generate_file(to_f, from_f, is_poem=True):
    with open(to_f, "w") as f:
        f.write(preprocess_text(from_f, is_poem))


def generate_book(to_f, from_f, is_poem=False):
    with open(to_f, "w", encoding="utf-8") as f:
        f.write(preprocess_text(from_f, is_poem))


def append_files(file_list, output_file, directory_path, is_poem):
    with open(output_file, "w") as outfile:
        for fname in file_list:
            file_path = os.path.join(directory_path, fname)
            text = preprocess_text(file_path, is_poem=is_poem)
            outfile.write(text)


def split_files(directory_path, train_output, test_output, is_poem):
    file_names = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    assert len(file_names) == 30, "There should be exactly 30 text files."
    train_files = file_names[:25]
    test_files = file_names[25:]
    append_files(train_files, train_output, directory_path, is_poem)
    append_files(test_files, test_output, directory_path, is_poem)


# # Define the recombine_syllables function
# def recombine_syllables(syllables, dictionary):
#     words = []
#     current_word = ''
#     for syllable in syllables:
#         potential_word = current_word + syllable
#         if potential_word in dictionary:
#             words.append(potential_word)
#             current_word = ''
#         else:
#             current_word = potential_word
#         if current_word:
#             words.append(current_word)
#     return ' '.join(words)


def main():
    if not os.path.exists("../data/train.txt"):
        generate_file(
            "../data/train.txt", "../utils/Shakespeare/Sonnets/THE SONNETS.txt"
        )

    if not os.path.exists("../data/test.txt"):
        generate_file(
            "../data/test.txt", "../utils/Shakespeare/Poems/A LOVER'S COMPLAINT.txt"
        )

    if not os.path.exists("../data/test_book.txt"):
        generate_book("../data/test_book.txt", "../data/adventures_test.txt")

    if not os.path.exists("../data/train_book.txt"):
        generate_book("../data/train_book.txt", "../data/adventures.txt")

    if not os.path.exists("../data/train_hyphen.txt"):
        generate_file(
            "../data/train_hyphen.txt", "../utils/Shakespeare/Sonnets/THE SONNETS.txt"
        )

    # directory_path = '../utils/Shakespeare/Plays'
    # split_files(directory_path, '../data/train_grammar.txt', '../data/test_grammar.txt', is_poem=False)


if __name__ == "__main__":
    main()
