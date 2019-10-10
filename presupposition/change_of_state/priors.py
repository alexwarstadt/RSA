import math
from presupposition.set_operations import *
from presupposition.change_of_state.worlds_and_utterances import *
import random
from presupposition.change_of_state.constants import *
import matplotlib.pyplot as plt
import numpy as np

# ==== Knowledgable priors ====

# # Speaker assumes listener shares some/most beliefs
# # k_l = 0.1
# def make_listener_context_prior(C_S):
#     def listener_context_prior_un_norm(C_S, C_L):
#         if not entails(C_S, C_L):
#             raise ValueError("C_S must entail C_L")
#         return math.pow(k_l, 1 - math.log(conditional_probability(C_S, C_L)))
#     probs = {}
#     for C_L in contexts:
#         try:
#             probs[frozenset(C_L)] = listener_context_prior_un_norm(C_S, C_L)
#         except ValueError:
#             continue
#     return normalize_probs(probs)
#
# # Listener assumes that speaker knows a lot
# # k_s = 1
# def make_speaker_context_prior(C_L):
#     def speaker_context_prior_un_norm(C_S, C_L):
#         if not entails(C_S, C_L):
#             raise ValueError("C_S must entail C_L")
#         # if len(C_L) == 1:
#         #     raise ValueError("C_L must contain uncertainty")
#         # Adding 1 means that whether or not the surprisal is > 1 does not determine whether or not the exponential explodes
#         # This might be unnecessary once normalization is taken into account
#         return math.pow(k_s, 1 - math.log(conditional_probability(C_S, C_L)))
#     probs = {}
#     for C_S in contexts:
#         try:
#             probs[frozenset(C_S)] = speaker_context_prior_un_norm(C_S, C_L)
#         except ValueError:
#             continue
#     return normalize_probs(probs)



# ====== Observation Priors ======


def make_common_grounds(observations):
    common_grounds = {frozenset({frozenset(worlds)}): 1}
    for obs in observations.keys():
        new_cgs = common_grounds.copy()
        for cg in common_grounds.keys():
            new_cgs[cg] = common_grounds[cg] * (1 - observations[obs])
            new_cgs[cg.union(frozenset({frozenset(literal_meaning(obs))}))] = common_grounds[cg] * (observations[obs]) * 0.5
            new_cgs[cg.union(frozenset({frozenset(literal_meaning("not_" + obs))}))] = common_grounds[cg] * (observations[obs]) * 0.5
        common_grounds = new_cgs
    return normalize_probs(common_grounds)

x = make_common_grounds(speaker_observations)

def make_speaker_context_prior(C_L):
    common_grounds = make_common_grounds(speaker_observations)
    context_prior = {}
    for cg in common_grounds:
        C_S = set.intersection(*[set(p) for p in cg])
        if entails(C_S, C_L):
            context_prior[frozenset(C_S)] = common_grounds[cg] + epsilon
    for C_S in contexts:
        if entails(C_S, C_L):
            if frozenset(C_S) not in context_prior.keys():
                context_prior[frozenset(C_S)] = epsilon
        else:
            context_prior[frozenset(C_S)] = 0
    return normalize_probs(context_prior)




def make_listener_context_prior(C_S):
    common_grounds = make_common_grounds(listener_observations)
    context_prior = {}
    for cg in common_grounds:
        C_L = set.intersection(*[set(p) for p in cg])
        if entails(C_S, C_L):
            context_prior[frozenset(C_L)] = common_grounds[cg] + epsilon
    for C_L in contexts:
        if entails(C_S, C_L):
            if frozenset(C_L) not in context_prior.keys():
                context_prior[frozenset(C_L)] = epsilon
        else:
            context_prior[frozenset(C_L)] = 0
    return normalize_probs(context_prior)





# def make_speaker_context_prior(C_L):
#     for obs in speaker_observations.keys():



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


# P_C_S = make_speaker_context_prior(speaker_observations, literal_meaning("null"))
# P_C_L = make_listener_context_prior(speaker_observations, literal_meaning("always"))

# P_C_Ss = {}
# for k_s in range(1,100):
#     try:
#         P_C_Ss[k_s] = make_speaker_context_prior(literal_meaning("null"))
#     except ZeroDivisionError:
#         pass

# P_C_Ls = {}
# for k_l in [x / 100 for x in range(100)]:
#     try:
#         P_C_Ls[k_l] = make_listener_context_prior(literal_meaning("always"))
#     except ZeroDivisionError:
#         pass
# vals = [speaker_context_prior[frozenset(worlds)][frozenset(c)] for c in contexts]
# context_names = [str(c) for c in contexts]
# y_pos = np.arange(len(contexts))
# plt.bar(y_pos, vals, align='center', alpha=0.5)
# plt.xticks(y_pos, context_names, rotation='vertical')
# plt.ylabel('Pr( C_S=___ | C_L=W)')
# plt.title('Speaker Context Prior, observation prior={1, 0.5}')
# plt.show()

pass