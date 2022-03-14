from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stops - are a set of commonly used words in a language. 
# Examples of stop words in English are a, the, is, are and etc. 
# Stop words are commonly used in Text Mining and Natural Language Processing 
# (NLP) to eliminate words that are so commonly used that 
# they carry very little useful information.



example_sentnce = "Watching Marcelo start a game brings me a great deal of pain and suffering. The history of a signle player, who has done a great deal, does not merit a strat in Real Madrid."
stop_words = set(stopwords.words("english"))


words = word_tokenize(example_sentnce)

filtered_sentence = []

for word in words:
    if word not in stop_words:
        filtered_sentence.append(word)
print(filtered_sentence)