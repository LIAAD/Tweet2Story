"""
    text2story/core/utils module

    Functions
    ---------
    chunknize_actors(annotations)
        converts the result from the 'extract_actors' method, implemented by the annotators supported, to a list the chunknized list of actors
        to do this conversion, the IOB in the NE tag is used
"""

from itertools import tee


def chunknize_actors(annotations):
    """
    Parameters
    ----------
    annotations : list[tuple[tuple[int, int], str, str]]
        list of annotations made by some tool for each token

    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        the list of actors identified where each actor is represented by a tuple
    """

    actors = []

    ready_to_add = False

    prev_ne_tag = ''

    for ann in annotations:
        token_character_span, token_pos_tag, token_ne_iob_tag = ann

        # token_ne_iob_tag = 'I-PER', then token_ne_iob_tag[2:] == 'PER'
        if token_ne_iob_tag.startswith("B") or (token_ne_iob_tag.startswith("I") and token_ne_iob_tag[2:] != prev_ne_tag):
            # Case we start a new chunk, after finishing another, for instance: Case B-Per, I-Per, B(or I)-Org, then we add the finished actor
            if ready_to_add:
                actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
                actors.append(actor)

            ready_to_add = True
            actor_start_offset = token_character_span[0]
            actor_end_offset = token_character_span[1]
            actor_lexical_head = token_pos_tag if token_pos_tag in ['Noun', 'Pronoun'] else 'UNDEF'
            actor_actor_type = token_ne_iob_tag[2:]

        elif token_ne_iob_tag.startswith("I"):
            # actor_start_offset it's always the same, since it's defined by the first token of the actor
            actor_end_offset = token_character_span[1]
            actor_lexical_head = actor_lexical_head if actor_lexical_head != 'UNDEF' else token_pos_tag if token_pos_tag in ['Noun', 'Pronoun'] else 'UNDEF'
            # actor_actor_type it's the same for all tokens that constitute the actor and it's already defined by the first token of the actor

        elif token_ne_iob_tag.startswith("O") and ready_to_add:
            actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
            actors.append(actor)
            ready_to_add = False
            # No need to reset the variables, since the first update writes over

        prev_ne_tag = token_ne_iob_tag[2:]

    if ready_to_add:  # If the last token still makes part of the actor
        actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
        actors.append(actor)

    return actors


def pairwise(iterable):
    """
    Iterate through some iterable with a lookahead.
    From the itertools docs recipes - https://docs.python.org/3/library/itertools.html
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
