"""
    AllenNLP annotator

    Used for:
        - Coref resolution
            'en' : 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
"""

import pandas as pd
import numpy as np
from itertools import zip_longest

from nltk.tokenize import sent_tokenize

from allennlp.predictors.predictor import Predictor

SRL_TYPE_MAPPING = {
    "TMP": "time",
    "LOC": "location",
    "ADV": "theme",  # Adverbial
    "MNR": "manner",
    "CAU": "cause",
    "EXT": "theme",  # should be attribute -> changed for compatibility
    "DIS": "theme",  # connection of two expressions -> should be theme -> changed for compatibility
    "PNC": "purpose",
    "PRP": "purpose",
    "NEG": "theme",  # should be attribute -> changed for compatibility | NEG = negation
    "DIR": "path",  # should be setting -> changed for compatibility
    "MOD": "instrument",  # MOD = Modal
    "PRD": "theme",  # Secondary predicate -> should be attribute -> changed for compatibility
    "ADJ": "theme",
    "COM": "agent",  # COM = Comitative -> used to express accompaniment
    "GOL": "goal",  # GOL = goal
    "REC": "instrument"  # REC = reciprocal
}

pipeline = {}

def load():
    pipeline['coref_en'] = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')
    pipeline["srl_en"] = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    )


def _normalize_sent_tags(sentence_df):
    """
    Normalize the frames retrieved from the SRL from one sentence.
    Each word must have only one label.

    @param sentence_df: DataFrame of the SRL each column is a word from the sentence and each row is the results of SRL
    for one frame.
    @return: List of the normalized tags
    @return: List of booleans of whether the tag is the beginning of the argument.
    """
    normalized_tags, begin_tags = [], []
    for col in np.arange(len(sentence_df.columns)):
        word_vals = sentence_df.iloc[:, col]

        word_vals = word_vals[word_vals != "O"]
        if word_vals.shape[0] == 1:
            normalized_tags.append(word_vals.iloc[0])
            begin_tags.append(word_vals.iloc[0].startswith("B"))
            continue
        verb_words = word_vals[word_vals.isin(["I-V", "B-V"])]
        if verb_words.shape[0] != 0:  # a) - verbo
            normalized_tags.append(verb_words.iloc[0])
            begin_tags.append(False)  # Event
            continue
        # b) - ARGM e ARG (o último e mais especifico tem prio)
        arg_words = word_vals[word_vals.str.contains(r".*[ARG][0-9]|ARGM")]
        if arg_words.shape[0] != 0:
            normalized_tags.append(arg_words.iloc[-1])  # desempate entre dois ARGM-X diferentes
            begin_tags.append(arg_words.iloc[-1].startswith("B"))
            continue
        else:
            print("\nNORMALIZATION ERROR - MULTIPLE TAG VALUES FOUND FOR WORD.")
            print(word_vals.values)

    return normalized_tags, begin_tags


def _find_events(normalized_tags, verb_tags, event_threshold=2):
    """
    Find words that belong to the same event.
    Each event can have a number of arguments between verbs and still be considered the same event.

    @param normalized_tags: result of normalized_sent_tags - a list of SRL tags for each word
    @param verb_tags: list of SRL tags for the algorithm to classify as event tags
    @param event_threshold: Threshold of non-verb arguments that can be found between verbs
    and still be considered part of the same event. n_args = event_threshold - 1
    @return: Boolean list of whether or not the word is part of an event
    """
    event_tags = []
    event_continue, event_begin = False, False
    for i, tag in zip_longest(np.arange(len(normalized_tags) - event_threshold), normalized_tags):
        if i is not None:
            if ("ARGM" in tag) & (normalized_tags[i + 1] in verb_tags):
                event_tags.append(True)
                event_begin = True
                continue

        if event_continue:
            event_tags.append(True)
        elif tag in verb_tags:
            event_tags.append(True)
            event_begin = True
        else:
            event_tags.append(False)

        if i is not None:
            conds = []
            for j in np.arange(1, event_threshold + 1):
                conds.append(normalized_tags[i + j] in verb_tags)

            if event_begin & any(conds):
                event_continue = True
            else:
                event_continue = False
                event_begin = False
        else:
            event_continue = False
            event_begin = False

    return event_tags


def _find_actors(begin_tags, event_tags):
    """
    Finds words that belong to the same actor or event and categorize them into actors.
    Each different argument represents a different actor. Arguments start at the begin tag (B).

    @param begin_tags: result of normalized_sent_tags - boolean list of tags that begin SRL arguments
    @param event_tags: result of find_events - boolean list of words that represent events
    @return: List of the actors that are represented by each word in the sentence
    """
    actor, i, actor_tags, event, is_event = False, 0, [], 1, False
    for btag, etag in zip(begin_tags, event_tags):
        if etag:
            is_event = True
        elif (not etag) & is_event:
            is_event = False
            event += 1

        if btag & (not etag):
            actor = True
            i += 1
            actor_tags.append("T" + str(i))
            continue

        if etag:
            actor_tags.append("EVENT" + str(event))
            actor = False
            continue

        if (not btag) & actor:
            actor_tags.append("T" + str(i))
        else:
            actor_tags.append(actor_tags[-1])  # In case everything else fails, just join with previous actor

    return actor_tags


def _srl_by_actor(srl_by_token, text, char_offset):
    """
    Organizes actors as words or expressions in the full text with their respective semantic role and character span.

    The semantic role is "EVENT" for event arguments and the SRL result for other actors.
    If the SRL result is a modifier, the most common in the actor is taken into account.

    The character span is a tuple - (start_char, end_char) -
    where the "start_char" is where the actor starts in the text and the "end_char" is where the actor ends.

    @param srl_by_token: DataFrame containing the results for the rest of the pipeline - namely, the actor references
    @param text: The full text to be annotated
    @param char_offset: The current char position in the text. Each word in the text increments it
    @return: Dict list with the actor, its semantic role and the its character position span in the text

    @note: CAN MALFUNCTION IF SRL DOES NOT FIND FRAMES IN EVERY SENTENCE.
    The char positions are checked in the entire text, if there is a word before in a sentence not recognized by the SRL
    it will malfunction slightly (give wrong position values).
    """
    result_list = []
    for actor in srl_by_token["actor"].unique():
        rows = srl_by_token[srl_by_token["actor"] == actor]
        tags = rows["tag"]
        if actor.startswith("EVENT"):
            sem_role_type = "EVENT"
        elif any("ARGM" in tag for tag in tags):
            argm_tags = [tag for tag in tags if "ARGM" in tag]
            sem_role_type = SRL_TYPE_MAPPING[argm_tags[0].split("-")[-1]]
        else:
            sem_role_type = "THEME"

        char_spans = []
        for token in rows.index:
            char_offset = text.find(token, char_offset)
            char_spans.append(char_offset)
            char_offset += len(token)

        result_list.append({
            "actor": ' '.join(rows.index), "sem_role_type": sem_role_type,
            "char_span": (char_spans[0], char_spans[-1] + len(token))
        })

    return result_list, char_offset


def _make_srl_df(text):
    """
    Make a pandas DataFrame with the results from the SRL for each sentence in the text.
    Each row of the DataFrame is a frame from the SRL.

    @param text: The full text to annotate
    @return: List of pandas DataFrames with the contents of the SRL (values) for each frame (row) by word (column)
    """
    sentences = sent_tokenize(text)

    srl = []
    for sent in sentences:
        result = pipeline['srl_en'].predict(sentence=sent)
        srl.append(result)

    dfs_by_sent = []
    for sentence in srl:
        tags_by_frame = pd.DataFrame(columns=sentence["words"])
        for i, frame in zip(np.arange(len(sentence["verbs"])), sentence["verbs"]):
            tags_by_frame.loc[i] = frame["tags"]

        if tags_by_frame.shape[0] != 0:
            dfs_by_sent.append(tags_by_frame)

    return dfs_by_sent


def _srl_pipeline(sentence_df, text, char_offset, verb_tags, event_threshold=3):
    """
    Pipeline to retrieve actors and events by their order in the text, with their semantic roles and character spans.
    Each iteration of the pipeline takes a dataframe of the SRL results for a sentence -> as returned by _make_srl_df.

    @param sentence_df: The DataFrame with the SRL results with the structure shown in _make_srl_df
    @param text: The full text to be annotated
    @param char_offset: The current character position offset to start looking for character spans (recursive)
    @param verb_tags: SRL tags that represent verbs -> defaults to ["B-V", "I-V"]
    @param event_threshold: Threshold of non-verb arguments that can be found between verbs
    and still be considered part of the same event. number_of_args = event_threshold - 1

    @return: df_by_actor -> Pandas DataFrame with each actor, their semantic roles and character spans
    character_offset -> The current character position offset in the full text
    """
    # 2. Remove words out of vocabulary in every frame #
    oov_mask = []
    for name, values in sentence_df.iteritems():
        oov_mask.append(any(~(values == "O")))
    sentence_df = sentence_df.loc[:, oov_mask]
    sentence_df = sentence_df.apply(lambda x: x.sort_values(ascending=False).values)  # Begin tags in the end always

    # 3. Normalize SRL tags for each word in the sentence dataframe #
    normalized_tags, begin_tags = _normalize_sent_tags(sentence_df)

    # 4. Define events #
    event_tags = _find_events(normalized_tags, verb_tags=verb_tags, event_threshold=event_threshold)

    # 5. Find and categorize expressions into actors #
    actor_tags = _find_actors(begin_tags, event_tags)

    # Putting together a dataframe for the characteristics of each word in this sentence
    df = pd.DataFrame({
        "tag": normalized_tags, "is_begin_tag": begin_tags, "is_event": event_tags, "actor": actor_tags
    }, index=sentence_df.columns)

    # 6. Find semantic roles and character spans for each actor in the sentence #
    df_by_actor, character_offset = _srl_by_actor(df, text, char_offset)

    return df_by_actor, character_offset


def extract_events(lang, text):
    """
    Main function that applies the SRL pipeline to extract event entities from each sentence.
    Joins every event actor from each sentence in the text.

    @param lang: The language of the text
    @param text: The full text to be annotated

    @return: Pandas DataFrame with every event entity and their character span
    """
    # 1. DATAFRAME WITH THE SRL RESULTS OF EVERY FRAME FOR EACH SENTENCE IN THE TEXT #
    dfs_by_sent = _make_srl_df(text)

    # FIND EVENTS - PIPELINE #
    character_offset, srl_actors_list = 0, []
    for sent_df in dfs_by_sent:
        df_by_actor, character_offset = _srl_pipeline(sent_df, text, character_offset, verb_tags=["B-V", "I-V"],
                                                      event_threshold=3)
        srl_actors_list.append(df_by_actor)

    srl_actors_list = [item for sublist in srl_actors_list for item in sublist]  # flatten list
    srl_df = pd.DataFrame(srl_actors_list)
    return srl_df[srl_df["sem_role_type"] == "EVENT"]


def extract_semantic_role_links(lang, text):
    """
    Main function that applies the SRL pipeline to extract the semantic role links between actors and events.
    Joins the Semantic Role Links from each sentence in the text.

    @param lang: The language of the text
    @param text: The full text to be annotated

    @return: List of pandas DataFrames that contains the SRL for each actor in each sentence.
    """
    dfs_by_sent = _make_srl_df(text)

    character_offset, srl_by_sentence = 0, []
    for sent_df in dfs_by_sent:
        df_by_actor, character_offset = _srl_pipeline(sent_df, text, character_offset, verb_tags=["B-V", "I-V"],
                                                      event_threshold=3)
        srl_by_sentence.append(pd.DataFrame(df_by_actor))

    return srl_by_sentence


def extract_objectal_links(lang, text):
    prediction = pipeline['coref_en'].predict(document=text)

    cluster_indexes_list = prediction["clusters"] # Indexes are token spans, we need character spans
    # Compute the character spans
    character_offset = 0
    character_span = []
    for token in prediction["document"]:
        character_offset = text.find(token, character_offset)
        character_span.append((character_offset, character_offset + len(token)))
        character_offset += len(token)

    # Convert the clusters to character spans
    for cluster in cluster_indexes_list:
        for i in range(len(cluster)):
            start_token_span = cluster[i][0]
            end_token_span = cluster[i][1]

            char_offset = (character_span[start_token_span][0], character_span[end_token_span][1])

            cluster[i] = char_offset

    return cluster_indexes_list
