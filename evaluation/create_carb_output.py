import pandas as pd
import numpy as np
from itertools import zip_longest, tee
from pathlib import Path
import os
import time

from allennlp.predictors.predictor import Predictor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", None)

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

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

ROOT_PATH = os.path.join(Path(__file__).parent)
TEST_FILES = os.path.join(ROOT_PATH, "CaRB", "data")
EVAL_DIR = os.path.join(ROOT_PATH, "CaRB", "system_outputs", "test")


# For testing purposes
def confidence(input_text):
    return 1.0/(1+0.005*len(input_text))

def pairwise(iterable):
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def normalize_sent_tags(sentence_df):
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


def find_events(normalized_tags, verb_tags, event_threshold=2):
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
            for j in np.arange(1, event_threshold+1):
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


def find_actors(begin_tags, event_tags):
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
            actor_tags.append("EVENT"+str(event))
            actor = False
            continue

        if (not btag) & actor:
            actor_tags.append("T" + str(i))
        else:
            if len(actor_tags) > 0:
                actor_tags.append(actor_tags[-1])  # In case everything else fails, just join with previous actor
            else:
                i += 1
                actor_tags.append("T" + str(i))

    return actor_tags


def srl_by_actor(srl_by_token, text, char_offset):
    result_list = []
    for actor in srl_by_token["actor"].unique():
        rows = srl_by_token[srl_by_token["actor"] == actor]
        tags = rows["tag"]
        if actor.startswith("EVENT"):
            sem_role_type = "EVENT"
        elif any(tag.endswith("ARG0") for tag in tags):
            sem_role_type = "AGENT"
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

    return pd.DataFrame(result_list), char_offset


if __name__ == '__main__':
    start = time.time()
    with open(os.path.join(TEST_FILES, "test.txt"), "r+") as f:
        tweets = f.read()
    f.close()

    sentences = tweets.split("\n")

    # SEMANTIC ROLE LABELLING #
    srl = []
    for sent in sentences:
        result = predictor.predict(
            sentence=sent
        )
        srl.append(result)

    # 1. DATAFRAME COM TODAS AS PALAVRAS E AS SUAS TAGS #
    dfs_by_sent = []
    for sentence in srl:
        tags_by_frame = pd.DataFrame(columns=sentence["words"])
        for i, frame in zip(np.arange(len(sentence["verbs"])), sentence["verbs"]):
            tags_by_frame.loc[i] = frame["tags"]

        if tags_by_frame.shape[0] != 0:
            dfs_by_sent.append(tags_by_frame)

    # NORMALIZE SRL DATAFRAME #
    f = open(os.path.join(EVAL_DIR, "tweet2story_output.txt"), "w", encoding="utf-8")
    character_offset = 0
    for sent_df in dfs_by_sent:
        # 2. Remove words out of vocabulary in every frame
        oov_mask = []
        for name, values in sent_df.iteritems():
            oov_mask.append(any(~(values == "O")))
        sent_df = sent_df.loc[:, oov_mask]
        sent_df = sent_df.apply(lambda x: x.sort_values(ascending=False).values)  # Begin tags in the end always

        # 3. Normalizar tags para cada palavra em cada frame
        normalized_tags, begin_tags = normalize_sent_tags(sent_df)

        # 4. JUNTAR VERBOS PRÓXIMOS (PARA DETERMINAR EVENTOS) #
        event_tags = find_events(normalized_tags, verb_tags=["B-V", "I-V"], event_threshold=3)

        # 5. IDENTIFICAR ATORES DE CADA ARGUMENTO #
        actor_tags = find_actors(begin_tags, event_tags)

        # Putting together a dataframe for the EVENTS on this sentence
        df = pd.DataFrame({
            "tag": normalized_tags, "is_begin_tag": begin_tags,
            "is_event": event_tags, "actor": actor_tags
        }, index=sent_df.columns)

        # 6. DATAFRAME BY ACTOR
        df_by_actor, character_offset = srl_by_actor(df, tweets, character_offset)

        for row1, row2, row3 in pairwise(df_by_actor.itertuples()):
            if row2.sem_role_type == "EVENT":
                input_sent = ' '.join(sent_df.columns)
                conf = confidence(input_sent)
                f.write(f"{input_sent}\t{str(conf)}\t{row2.actor}\t{row1.actor}\t{row3.actor}\n")

    f.close()

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")
