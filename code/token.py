import re
import pyphen

file_path = "../utils/Shakespeare/Poems/A LOVER'S COMPLAINT.txt"
def preprocess_text(file_path):
    dic = pyphen.Pyphen(lang='en')

# Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = text.lower()
        text = text.replace("'''", "'")
        text = text.replace('\n', ' <br> ') + ' <STOP>'
        text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'\b[A-Z][a-z]*\b', '<UNK>', text)
        text = ' <SEP> '.join(text.split())

        # Hyphenate 
        words = text.split()
        hyphenated_text = ' '.join([dic.inserted(word, hyphen=' ') for word in words])
        verses = hyphenated_text.split('<SEP> <br> <SEP> <br>')  # <SEP> <br> <SEP> <br> indicate an empty line / verse break
        hyphenated_text_with_verses = ' <VER> '.join(verses)

        return hyphenated_text_with_verses

preprocessed_text = preprocess_text(file_path)

with open('poems_verses.txt', 'w', encoding='utf-8') as file:
    file.write(preprocessed_text)

