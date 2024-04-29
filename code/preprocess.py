
import os
import tensorflow as tf
import numpy as np
from functools import reduce
import nltk
#nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# import syllapy
# from hyphenate import hyphenate_word
import re

plays = [
"All's Well That Ends Well", "ALLS WELL THAT ENDS WELL", "Antony and Cleopatra", "As You Like It",
"The Comedy of Errors", "Cymbeline", "Love's Labours Lost",
"Measure for Measure", "The Merry Wives of Windsor", "The Merchant of Venice",
"A Midsummer Night's Dream", "Much Ado About Nothing", "Pericles, Prince of Tyre",
"Taming of the Shrew", "The Tempest", "Troilus and Cressida", "Twelfth Night",
"Two Gentlemen of Verona", "Winter's Tale", "Henry IV, part 1", "Henry IV, part 2",
"Henry V", "THE FIRST PART OF KING HENRY THE FOURTH", "Henry VI, part 1", "Henry VI, part 2", "Henry VI, part 3",
"Henry VIII", "King John", "Richard II", "Richard III", "Antony and Cleopatra",
"Coriolanus", "Hamlet", "Julius Caesar", "King Lear", "Macbeth", "Othello",
"Romeo and Juliet", "Timon of Athens", "Titus Andronicus", "THE TRAGEDY OF KING LEAR", 
"THE LIFE OF KING HENRY THE FIFTH","THE FIRST PART OF HENRY THE SIXTH", "THE THIRD PART OF KING HENRY THE SIXTH", 
"KING HENRY THE EIGHTH", "LOVE'S LABOUR'S LOST",
'SECOND PART OF KING HENRY IV', "THE SECOND PART OF KING HENRY THE SIXTH"]

sonnets = [
"1-17", "18-126", "127-154", "THE SONNETS"
]

poems = [
"A LOVER'S COMPLAINT", "PASSIONATE PILGRIM", "PHOENIX AND THE TURTLE",
"RAPE OF LUCRECE", "VENUS AND ADONIS", "FUNERAL ELEGY BY W.S."
]

poems = [title.upper() for title in poems]
sonnets = [title.upper() for title in sonnets]
plays = [title.upper() for title in plays]

os.makedirs('..utils/Shakespeare/Plays', exist_ok=True)
os.makedirs('..utils/Shakespeare/Sonnets', exist_ok=True)
os.makedirs('..utils/Shakespeare/Poems', exist_ok=True)
os.makedirs('..utils/Shakespeare/Other', exist_ok=True)


def get_title_above_separator(text, separator):
    pattern = re.compile(r'^(.*)(?:\n|\r\n?)(?:.*)(?:\n|\r\n?)' + re.escape(separator), re.MULTILINE)
    titles = pattern.findall(text)
    return titles

def split_shakespeare():
    with open('..\data\shakespeare.txt', 'r') as file:
        content = file.read()

    separator = "by William Shakespeare"

    titles = get_title_above_separator(content, separator)

    for title in titles:

        print(title)

    def save_text(category, title, text):
        with open(f'Shakespeare/{category}/{title}.txt', 'w') as file:
            file.write(text)


    sections = content.split('by William Shakespeare')

    for section in sections:
        for title in plays:
            if title in section:
                save_text('Plays', title, section)
                break
        for title in sonnets:
            if title in section:
                save_text('Sonnets', title, section)
                break
        for title in poems:
            if title in section:
                save_text('Poems', title, section)
                break


# def make_poem_ready(fname):
#     with open(fname, "r") as file:
#         lines = file.readlines()
    
#     tokens = [word_tokenize(line) for line in lines]

#     full_tokens = [token for sublist in tokens for token in sublist]

#     #print(full_tokens)

#     hyphen_toknes = []
    
#     with open("sample.txt", "w") as f:
#         count = 0
#         for token in full_tokens:
#             count += 1
#             #print(token)

#             for hyph in hyphenate_word(token):
#                 f.write(" " + hyph + " ")
#             f.write("<SEP>")
            
#             if count % 10 == 0:
#                 f.write("\n")
            
    
#     return hyphen_toknes

#stuff = make_poem_ready("Shakespeare/Poems/A LOVER'S COMPLAINT.txt")

#print(stuff)

### Get_data assumes that the files are of the format where every word is stemmed
### and <UNK> tokens are already present

# in our case, we can also hyphenate the file, since we really only need to do it once

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg index->word mapping)
    """
    # Hint: You might not use all of the initialized variables depending on how you implement preprocessing. 
    #vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.

    # with open(train_file, "r") as t_file:
    #     vocabulary = {'<UNK>': 0}
    #     vocab_index = 1  # Start indexing from 1
    #     for sentence in t_file:
    #         split_sen = sentence.lower().strip().split(" ")
    #         # print(split_sen)
    #         train_data += split_sen
    #         for word in split_sen:
    #             if word not in vocabulary:
    #                 vocabulary[word] = vocab_index
    #                 vocab_index += 1
    #                 train_data += split_sen
            
    #     file_unique_words = set(train_data)
    #     vocabulary = {w:i for i,w in enumerate(file_unique_words)}
    #     vocabulary["<UNK>"] = len(file_unique_words)
    
    # with open(test_file, "r") as test_f:
    #     for sentence in test_f:
    #         test_data += sentence.lower().strip().split(" ")
    #     test_data = [word if word in vocabulary else '<UNK>' for word in test_data]


    # # for word in test_data:
    # #     if word not in vocabulary:
    # #         print(word)
    # # Sanity Check, make sure there are no new words in the test data.
    # assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    # train_data = list(map(lambda x: vocabulary[x], train_data))
    # test_data  = list(map(lambda x: vocabulary[x], test_data))

    # # print("train_data", train_data)
    # return train_data, test_data, vocabulary


    vocabulary, train_data, test_data = {'<UNK>': 0}, [], []
    vocab_index = 1  # Start indexing from 1

# Process training data and build vocabulary
    with open(train_file, "r") as t_file:
        for sentence in t_file:
            split_sen = sentence.lower().strip().split(" ")
            for word in split_sen:
                if word not in vocabulary:
                    vocabulary[word] = vocab_index
                    vocab_index += 1
            train_data.extend(split_sen)

    with open(test_file, "r") as test_f:
        for sentence in test_f:
            split_sen = sentence.lower().strip().split(" ")
            test_data += [word if word in vocabulary else '<UNK>' for word in split_sen]

    assert all(word in vocabulary for word in test_data)

    train_data = [vocabulary[word] for word in train_data]
    test_data = [vocabulary[word] for word in test_data]

    return train_data, test_data, vocabulary
