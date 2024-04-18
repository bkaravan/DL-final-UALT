#### This is a general utils file 

## For now, dumping every function that looks potentially useful from source

#### HYPHENATION MODULE EXAMPLES

# from hyphenate import hyphenate_word

# #implements hyphenate to individual token
# def short_hyphenate(word):
#     h1= hyphenate_word(word) #implements hyphenate to word
#     t1= [ t for t in h1 if len(t)>0 ] #remove empty tokens from vocab 
#     return t1 


# # used for production and the prelude 
# #implements hyphenate to a list of tokens
# def get_hyp_lm_verses_works(verses): 
#     """
#     Hyphenates a list of verses 
#     :param verses: list of strings

#     :return: 
#     :new_verses: list of hyphenated verses with SEP tokens between words 
    
#     """
    
#     new_verses = []
#     for verse in verses:
#         new_verses.append([])  
#         for line in verse:
#             new_verses[-1].append([])
#             for hyp_w in line:
#                 processed_word=short_hyphenate(hyp_w) #hyphenates using hyphenate_word library

#                 new_verses[-1][-1].extend(processed_word)
 
#                 new_verses[-1][-1].append('<SEP>')# SEP tokens are added inbetween words
   
#             new_verses[-1][-1] = new_verses[-1][-1][:-1]

#     return new_verses

# #used for the Guide 
# #implements hyphenate to a list of tokens
# def get_hyp_lm_verses_guide(verses): 
#     """
#     Hyphenates a list of verses 
#     :param verses: list of strings

#     :return: 
#     :new_verses: list of hyphenated verses with SEP tokens between words 
    
#     """
#     new_verses = []
#     for line in verses:
#         new_verses.append([])
#         for hyp_w in line:
#             processed_word=short_hyphenate(hyp_w) #hyphenates using hyphenate_word library

#             new_verses[-1].extend(processed_word)

#             #print(f"currently: {pro}\n")

#             new_verses[-1].append('<SEP>') # SEP tokens are added inbetween words
   
#         new_verses[-1] = new_verses[-1][:-1]

#     return new_verses


# example = [["this", "is", "verse", "1", "of", "stuff"], 
#            ["and", "this", "is", "indeed", "the" ,"second"]]

# example2 = ["this is verse 1 of stuff", 
#            "and this is indeed the second", 
#            "blah blah, bluh bluh hey",
#            "there is a way for us to do DL!"]

# Create a hyphenator for a specific language, for example, Englis

def hyphenate_word(word):
    """
    Simple hyphenation function for English words.
    
    :param word: string, the word to hyphenate
    :return: list, hyphenated syllables
    """
    vowels = 'aeiouy'
    
    # Initialize variables
    syllables = []
    current_syllable = ''
    
    for char in word:
        current_syllable += char
        
        # If the current character is a vowel, add the current syllable to the list
        if char in vowels:
            syllables.append(current_syllable)
            current_syllable = ''
    
    # If there's a remaining syllable, add it to the list
    if current_syllable:
        syllables.append(current_syllable)
    
    return syllables

# Example words to hyphenate
words = ["example", "hyphenation", "python", "neural", "network"]

# Hyphenate each word and print the results
for word in words:
    syllables = hyphenate_word(word)
    #print(f"{word}: {'-'.join(syllables)}")
    print(syllables)
