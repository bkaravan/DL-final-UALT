
import os
import tensorflow as tf
import numpy as np
from functools import reduce
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import syllapy

plays = [
"All's Well That Ends Well", "Antony and Cleopatra", "As You Like It",
"The Comedy of Errors", "Cymbeline", "Love's Labours Lost",
"Measure for Measure", "The Merry Wives of Windsor", "The Merchant of Venice",
"A Midsummer Night's Dream", "Much Ado About Nothing", "Pericles, Prince of Tyre",
"Taming of the Shrew", "The Tempest", "Troilus and Cressida", "Twelfth Night",
"Two Gentlemen of Verona", "Winter's Tale", "Henry IV, part 1", "Henry IV, part 2",
"Henry V", "Henry VI, part 1", "Henry VI, part 2", "Henry VI, part 3",
"Henry VIII", "King John", "Richard II", "Richard III", "Antony and Cleopatra",
"Coriolanus", "Hamlet", "Julius Caesar", "King Lear", "Macbeth", "Othello",
"Romeo and Juliet", "Timon of Athens", "Titus Andronicus", "THE TRAGEDY OF KING LEAR"
]

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

os.makedirs('Shakespeare/Plays', exist_ok=True)
os.makedirs('Shakespeare/Sonnets', exist_ok=True)
os.makedirs('Shakespeare/Poems', exist_ok=True)


import re

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


def make_poem_ready(fname):
    with open(fname, "r") as file:
        lines = file.readlines()
    
    tokens = [word_tokenize(line) for line in lines]

    full_tokens = [token for sublist in tokens for token in sublist]

    #print(full_tokens)

    hyphen_toknes = []
    
    with open("sample.txt", "w") as f:
        count = 0
        for token in full_tokens:
            count += 1
            print(token)
            syllables = syllapy.count(token)
            if syllables:
                hyphened = [token[i:i + syllables] for i in range(0, len(token), syllables)]
            
                for hyph in hyphened:
                    print(hyph)
                    f.write(hyph + " ")
            
            if count % 10 == 0:
                f.write("\n")
            
    
    return hyphen_toknes

stuff = make_poem_ready("Shakespeare/Poems/A LOVER'S COMPLAINT.txt")

print(stuff)

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
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.

    with open(train_file, "r") as t_file:
        for sentence in t_file:
            split_sen = sentence.lower().strip().split(" ")
            # print(split_sen)
            train_data += split_sen
        file_unique_words = set(train_data)
        vocabulary = {w:i for i,w in enumerate(file_unique_words)}
    
    with open(test_file, "r") as test_f:
        for sentence in test_f:
            test_data += sentence.lower().strip().split(" ")

    # for word in test_data:
    #     if word not in vocabulary:
    #         print(word)
    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    return train_data, test_data, vocabulary