
import os

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
