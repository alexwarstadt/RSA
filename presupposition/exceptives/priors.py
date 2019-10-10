from presupposition.set_operations import *
import math
from presupposition.exceptives.worlds_and_utterances import *
from presupposition.exceptives.constants import *

# ============ CONTEXT PRIORS ============

# ==== Uniform priors ====

# # uniform over entailed contexts
# def listener_context_prior(C_S):
#     probs = {}
#     entailed_contexts = filter(lambda C_L: entails(C_S, C_L) and not entails(C_L, C_S), contexts)
#     for C_L in entailed_contexts:
#         probs[frozenset(C_L)] = 1
#     return normalize_probs(probs)

# Gives uniform distribution over singleton contexts
# def make_speaker_context_prior(C_L):
#     if len(C_L) == 1:
#         raise ValueError("C_L must contain uncertainty")
#     probs = {}
#     for C_S in contexts:
#         if len(C_S) != 1:
#             continue
#         if not entails(C_S, C_L):
#             continue
#         probs[frozenset(C_S)] = 1
#     return normalize_probs(probs)


# ==== Observation priors ====

# k_l = 0.5
# def make_listener_context_prior(C_S):
#

# ==== Knowledgable priors ====

# Speaker assumes listener shares some/most beliefs
def make_listener_context_prior(C_S):
    def listener_context_prior_un_norm(C_S, C_L):
        if not entails(C_S, C_L):
            raise ValueError("C_S must entail C_L")
        # if len(C_L) == 1:
        #     raise ValueError("C_L must contain uncertainty")
        # Adding 1 means that whether or not the surprisal is > 1 does not determine whether or not the exponential explodes
        # This might be unnecessary once normalization is taken into account
        return math.pow(k_l, math.log(conditional_probability(C_S, C_L)))
    probs = {}
    for C_L in contexts:
        try:
            probs[frozenset(C_L)] = listener_context_prior_un_norm(C_S, C_L)
        except ValueError:
            continue
    return normalize_probs(probs)

# Listener assumes that speaker knows a lot
def make_speaker_context_prior(C_L):
    def speaker_context_prior_un_norm(C_S, C_L):
        if not entails(C_S, C_L):
            raise ValueError("C_S must entail C_L")
        # if len(C_L) == 1:
        #     raise ValueError("C_L must contain uncertainty")
        # Adding 1 means that whether or not the surprisal is > 1 does not determine whether or not the exponential explodes
        # This might be unnecessary once normalization is taken into account
        return math.pow(k_s, 1 - math.log(conditional_probability(C_S, C_L)))
    probs = {}
    for C_S in contexts:
        try:
            probs[frozenset(C_S)] = speaker_context_prior_un_norm(C_S, C_L)
        except ValueError:
            continue
    return normalize_probs(probs)


# ==== build priors ====

listener_context_prior = {}
for C_S in contexts:
    try:
        listener_context_prior[frozenset(C_S)] = make_listener_context_prior(C_S)
    except ValueError:
        continue

speaker_context_prior = {}
for C_L in contexts:
    try:
        speaker_context_prior[frozenset(C_L)] = make_speaker_context_prior(C_L)
    except ValueError:
        continue


P_C_Ss = {}
for k_s in range(1,100):
    try:
        P_C_Ss[k_s] = make_speaker_context_prior(literal_meaning("null"))
    except ZeroDivisionError:
        pass

import pylab as plt
[{(1,1,1)}, {(1,1,1), (1,1,0)}]


pass