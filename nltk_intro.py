from doctest import Example
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# nltk.download()
# tokenizing - word tokenizers.... sentence tokenizers
# corpora - body of text. ex: medical journals, presidential speeches, English language
# lexicon - words and their meaning


Example_text = "Girl I feel too. Let it be, baby breath. I swear i'm right here. Do you wanna be high for this?"
print(sent_tokenize(Example_text))
print(word_tokenize(Example_text))