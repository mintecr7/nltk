# stemming is the process of reducing inflected words to their word stem, 
# base or root form generally a written word form.

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

exapmle_words = ["python", "pythoner", "pythoning", "pyhthoned", "pythonly"]

stem = [ps.stem(w) for w in exapmle_words]


# print(stem)

new_text = "It is very important to be pyhtonly while you are pythoning with python. All pythoners have pythoned poorly atleast once."
words = word_tokenize(new_text)
stemmed = [ps.stem(w) for w in words ]

print(stemmed)