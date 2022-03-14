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

# chinking -  removing something from a chunk. collect everything except onething 


import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union


training_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = PunktSentenceTokenizer(train_text=training_text)
tokenized = custom_sentence_tokenizer.tokenize(text=sample_text)


def process_content():
    try:
        for token in tokenized:
            words  = nltk.word_tokenize(token)
            pos = nltk.pos_tag(words)
            chunk_gram = r"""chunk: {<.*>+}
                                    }<VB.?|IN|DT>+{"""
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chinked = chunk_parser.parse(pos)
            chinked.draw()
            
    except Exception as error:
        print("fara fix this:- {}".format(error))

process_content()


"""
{1,3} = for digits, u expect 1-3 counts of digits, or "places"
. = anything 
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions
$ = matches at the end of string
^ = matches start of a string
| = matches either/or. Example x|y = will match either x or y
[] = range, or "variance"
{x} = expect to see this amount of the preceding code.
{x,y} = expect to see this x-y amounts of the precedng code
"""