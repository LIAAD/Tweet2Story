'''
    SPARKNLP annotator

    Used for actor extraction
        'pt' : wikiner_6B_100
        'en' : default
'''

from text2story.core.utils import chunknize_actors
from text2story.core.exceptions import InvalidLanguage

import sparknlp
from pyspark.sql import SparkSession
from sparknlp.base import DocumentAssembler, LightPipeline
from sparknlp.annotator import Tokenizer, PerceptronModel, WordEmbeddingsModel, NerDLModel, NerCrfModel
from pyspark.ml import Pipeline
import pandas as pd

pipeline = {}

def load():
    """
    Used, at start, to load the pipeline for the supported languages.
    """

    sparknlp.start()
    spark = SparkSession.builder.appName("t2s").getOrCreate()
    spark.sparkContext.setLogLevel("FATAL")

    documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

    tokenizer         = Tokenizer().setInputCols(["document"]).setOutputCol("token")

    embeddings        = WordEmbeddingsModel.pretrained('glove_100d').setInputCols(["token", "document"]).setOutputCol("embeddings")

    pos_tagger_pt     = PerceptronModel.pretrained('pos_ud_bosque', 'pt').setInputCols(["token", "document"]).setOutputCol("pos")
    pos_tagger_en     = PerceptronModel.pretrained('pos_anc', 'en').setInputCols(["token", "document"]).setOutputCol("pos")

    ner_model_pt      = NerDLModel.pretrained('wikiner_6B_100', 'pt').setInputCols(["document", "token", "embeddings"]).setOutputCol("ner")
    ner_model_en      = NerCrfModel.pretrained().setInputCols(["document", "token", "pos", "embeddings"]).setOutputCol("ner")

    pipeline_pt = Pipeline(stages=[documentAssembler, tokenizer, embeddings, pos_tagger_pt, ner_model_pt])
    pipeline['pt'] = LightPipeline(pipeline_pt.fit(spark.createDataFrame(pd.DataFrame({'text': ['']}))))

    pipeline_en = Pipeline(stages=[documentAssembler, tokenizer, embeddings, pos_tagger_en, ner_model_en])
    pipeline['en'] = LightPipeline(pipeline_en.fit(spark.createDataFrame(pd.DataFrame({'text': ['']}))))


def extract_actors(lang, text):
    """
    Parameters
    ----------
    lang : str
        the language of text to be annotated
    text : str
        the text to be annotated
    
    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        the list of actors identified where each actor is represented by a tuple

    Raises
    ------
        InvalidLanguage if the language given is invalid/unsupported
    """

    if lang not in ['pt', 'en']:
        raise InvalidLanguage

    doc = pipeline[lang].fullAnnotate(text)[0] 

    iob_token_list = []
    for i in range(len(doc['token'])):
        start_char_offset = doc['token'][i].begin
        end_char_offset   = doc['token'][i].end + 1
        char_span         = (start_char_offset, end_char_offset)
        pos_tag           = normalize(doc['pos'][i].result)
        ne                = doc['ner'][i].result[:2] + normalize(doc['ner'][i].result[2:]) if doc['ner'][i].result[:2] != 'O' else 'O'

        iob_token_list.append((char_span, pos_tag, ne))

    actor_list = chunknize_actors(iob_token_list)

    return actor_list


def normalize(label):
    """
    Parameters
    ----------
    label : str
    
    Returns
    -------
    str
        the label normalized
    """

    mapping = {
        # POS tags 
        # en: (Penn Treebank Project: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
        'NN'    : 'Noun',
        'NNS'   : 'Noun',
        'NNP'   : 'Noun',
        'NNPS'  : 'Noun',
        'PRP'   : 'Pronoun',
        'PRP$'  : 'Pronoun',
        'WP'    : 'Pronoun',
        'WP$'   : 'Pronoun',
    
        # pt: Universal POS Tags: http://universaldependencies.org/u/pos/
        #"ADJ": "adjective",
        #"ADP": "adposition",
        #"ADV": "adverb",
        #"AUX": "auxiliary",
        #"CONJ": "conjunction",
        #"CCONJ": "coordinating conjunction",
        #"DET": "determiner",
        #"INTJ": "interjection",
        "NOUN": "Noun",
        #"NUM": "numeral",
        #"PART": "particle",
        "PRON": "Pronoun",
        "PROPN": "Noun",
        #"PUNCT": "punctuation",
        #"SCONJ": "subordinating conjunction",
        #"SYM": "symbol",
        #"VERB": "verb",
        #"X": "other",
        #"EOL": "end of line",
        #"SPACE": "space",

        # NE labels
        'LOC'   : 'Loc',
        'ORG'   : 'Org',
        'PER'   : 'Per',
        'MISC'  : 'Other'    
    }

    return mapping.get(label, 'UNDEF')
