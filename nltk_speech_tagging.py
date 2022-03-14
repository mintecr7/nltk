# part-of-speech tagging (POS tagging or PoS tagging or POST), 
# also called grammatical tagging is the process of marking up a word in a text 
# (corpus) as corresponding to a particular part of speech,
# based on both its definition and its context. A simplified 
# form of this is commonly taught to school-age children, 
# in the identification of words as nouns, verbs, adjectives, adverbs, etc.

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer 
# PunktSentenceTokenizer - is unsupervised machine learning sentence tokenizer


training_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
# print(sample_text)

custom_sentence_tokenizer = PunktSentenceTokenizer(training_text)

tokenized = custom_sentence_tokenizer.tokenize(sample_text)

def process_content():
    taggeds = []
    try:
        for word in tokenized:
            words = nltk.word_tokenize(word)
            tagged = nltk.pos_tag(words)
            taggeds.append(tagged)

        return taggeds
    except Exception as error:
        print(str(error))
    

tagged = process_content()
print(tagged)

# Abbreviation	Meaning
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	This NLTK POS Tag is an adjective (large)
# JJR	adjective, comparative (larger)
# JJS	adjective, superlative (largest)
# LS	list market
# MD	modal (could, will)
# NN	noun, singular (cat, tree)
# NNS	noun plural (desks)
# NNP	proper noun, singular (sarah)
# NNPS	proper noun, plural (indians or americans)
# PDT	predeterminer (all, both, half)
# POS	possessive ending (parent\ 's)
# PRP	personal pronoun (hers, herself, him,himself)
# PRP$	possessive pronoun (her, his, mine, my, our )
# RB	adverb (occasionally, swiftly)
# RBR	adverb, comparative (greater)
# RBS	adverb, superlative (biggest)
# RP	particle (about)
# TO	infinite marker (to)
# UH	interjection (goodbye)
# VB	verb (ask)
# VBG	verb gerund (judging)
# VBD	verb past tense (pleaded)
# VBN	verb past participle (reunified)
# VBP	verb, present tense not 3rd person singular(wrap)
# VBZ	verb, present tense with 3rd person singular (bases)
# WDT	wh-determiner (that, what)
# WP	wh- pronoun (who)
# WRB	wh- adverb (how)
