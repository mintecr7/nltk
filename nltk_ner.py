"""
Named-entity recognition (NER) (also known as (named) entity identification, 
entity chunking, and entity extraction) is a subtask of information extraction 
that seeks to locate and classify named entities mentioned in unstructured text 
into pre-defined categories such as person names, organizations, locations, 
medical codes, time expressions, quantities, monetary values, percentages, etc.
"""
from random import sample
import nltk 
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union
from pyparsing import Word



train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

sent_tokenizer = PunktSentenceTokenizer(train_text=train_text)
tokenized = sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for token in tokenized: 
            # tokenize the words from the each given sentence
            words = word_tokenize(token)

            # speech tag each word as NNP, DT ......
            pos = nltk.pos_tag(words)
            
            # Take out named entities from the pos list
            NER = nltk.ne_chunk(pos)
            
            # draw the named entities
            NER.draw()

        pass
    except Exception as error:
        print("fara fix this issue :- {}".format(error))


process_content()

"""
NE types   Examples
ORGANIZATION Korean insitute of science and technology
PERSON Chris Brown, President Fara
LOCATION  Lake Hawassa, Mount Tabor
DATA  june, 2017-09-23
TIME two fifty am, 2:32 p.m.
MONEY  6000birr, BGP 10.45, 
PERCENT twenty pct, 34.9%
FACILITY Washington Monumnet, Axum
GPE   South East asia, East Africa 
"""
