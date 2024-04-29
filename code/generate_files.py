import re
import pyphen
import os

#file_path = "../utils/Shakespeare/Poems/A LOVER'S COMPLAINT.txt"
def preprocess_text(file_path, is_poem=True):
    dic = pyphen.Pyphen(lang='en')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = text.lower()
        text = text.replace("'''", "'")
        text = text.replace('\n', ' <br> ') + ' <STOP>'
        text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'\b[A-Z][a-z]*\b', '<UNK>', text)
        text = ' <SEP> '.join(text.split())
        words = text.split()
        text = ' '.join([dic.inserted(word, hyphen=' ') for word in words])
        # Hyphenate 
        if is_poem:
            verses = text.split('<SEP> <br> <SEP> <br>')  # <SEP> <br> <SEP> <br> indicate an empty line / verse break
            text = ' <VER> '.join(verses)

        return text

#preprocessed_text = preprocess_text(file_path)

#with open('poems_verses.txt', 'w', encoding='utf-8') as file:
#    file.write(preprocessed_text)

def generate_train():
    with open('../data/train.txt', 'w') as f:
        f.write(preprocess_text("../utils/Shakespeare/Sonnets/THE SONNETS.txt"))

def generate_test():
    with open('../data/test.txt', 'w') as f:
        f.write(preprocess_text("../utils/Shakespeare/Poems/A LOVER'S COMPLAINT.txt"))

def append_files(file_list, output_file, directory_path, is_poem):
    with open(output_file, 'w') as outfile:
        for fname in file_list:
            file_path = os.path.join(directory_path, fname)
            text = preprocess_text(file_path, is_poem=is_poem)
            outfile.write(text)


def split_files(directory_path, train_output, test_output, is_poem):
    file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    assert len(file_names) == 30, "There should be exactly 30 text files."
    train_files = file_names[:25]
    test_files = file_names[25:]
    append_files(train_files, train_output, directory_path, is_poem)
    append_files(test_files, test_output, directory_path, is_poem)

directory_path = '../utils/Shakespeare/Plays'
split_files(directory_path, '../data/train_grammar.txt', '../data/test_grammar.txt', is_poem=False)


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
    if not os.path.exists('../data/train.txt'):
        generate_train()

    if not os.path.exists('../data/test.txt'):
        generate_test()

if __name__ == "__main__":
    main()