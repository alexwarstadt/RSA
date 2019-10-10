from presupposition.set_operations import *

def make_worlds():
    return {(1,1),
            (1,0),
            (0,1),
            (0,0)
            }

def make_utterances():
    simple_utterances = ["present",
                         "past",
                         "always",
                         "never",
                         "stop",
                         "start"]
    neg_utterances = ["not_" + u for u in simple_utterances]
    return simple_utterances + neg_utterances + ["null"]


def make_literal_meaning(worlds):
    def literal_meaning(u):
        if u == "present":
            return {(1,1), (0,1)}
        if u == "past":
            return {(1,1), (1,0)}
        if u == "always":
            return {(1,1)}
        if u == "never":
            return {(0,0)}
        if u == "stop":
            return {(1,0)}
        if u == "start":
            return {(0,1)}
        if u.startswith("not"):
            negation = u.split("_")[1]
            return set(filter(lambda w: not is_true_at(literal_meaning(negation), w), worlds))
        if u == "null":
            return worlds
    return literal_meaning


worlds = make_worlds()
contexts = powerset(worlds)
utterances = make_utterances()
literal_meaning = make_literal_meaning(worlds)


