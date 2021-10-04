from text2story.core.exceptions import InvalidTool
from text2story.annotators import SPACY, NLTK, SPARKNLP, PY_HEIDELTIME, ALLENNLP

ACTOR_EXTRACTION_TOOLS = ['spacy', 'nltk', 'sparknlp']
TIME_EXTRACTION_TOOLS = ['py_heideltime']
EVENT_EXTRACTION_TOOLS = ['allennlp']
OBJECTAL_LINKS_RESOLUTION_TOOLS = ['allennlp']
SEMANTIC_ROLE_LABELLING_TOOLS = ['allennlp']

def load():
    SPACY.load()
    NLTK.load()
    SPARKNLP.load()
    PY_HEIDELTIME.load()
    ALLENNLP.load()

def extract_actors(tool, lang, text):
    if tool == 'spacy':
        return SPACY.extract_actors(lang, text)
    elif tool == 'nltk':
        return NLTK.extract_actors(lang, text)
    elif tool == 'sparknlp':
        return SPARKNLP.extract_actors(lang, text)

    raise InvalidTool


def extract_times(tool, lang, text, publication_time):
    if tool == 'py_heideltime':
        return PY_HEIDELTIME.extract_times(lang, text, publication_time)

    raise InvalidTool


def extract_objectal_links(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_objectal_links(lang, text)

    raise InvalidTool


def extract_events(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_events(lang, text)

    raise InvalidTool


def extract_semantic_role_links(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_semantic_role_links(lang, text)

    raise InvalidTool
