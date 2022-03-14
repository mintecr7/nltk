from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# synset
# print(syns)
# print(syns[0].name())

# lemma
# print(syns[0].lemmas())

# just the word
# print(syns[0].lemmas()[0].name())

# definition
# print(syns[0].definition())

# examples 
# print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for lemm in syn.lemmas():
        synonyms.append(lemm.name())
        if lemm.antonyms():
            antonyms.append(lemm.antonyms()[0].name())
    

# print(synonyms)
# print(antonyms)

word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("boat.n.01")
print(word1.wup_similarity(word2))

word1 = wordnet.synset("football.n.01")
word2 = wordnet.synset("soccer.n.01")
print(word1.wup_similarity(word2))

word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("lotion.n.01")
print(word1.wup_similarity(word2))
