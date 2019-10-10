import math
from presupposition.change_of_state.worlds_and_utterances import *
from presupposition.change_of_state.priors import *
from presupposition.set_operations import *
import matplotlib.pyplot as plt
import numpy as np




# ============ Literal Listener =============
# Version 2: L0(C_S | u, C_L) =~ P(C_S | C_L & u)

def L0_un_norm(C_S, u, C_L):
    """version 2: L0(C_S | u, C_L) =~ d(C_S entails C_L & u) * P(C_S)"""
    u_m = literal_meaning(u)
    if not entails(C_S, intersect(u_m, C_L)):
        return 0
    else:
        return speaker_context_prior[frozenset(C_L)][frozenset(C_S)]

def L0_norm(u, C_L):
    if len(intersect(literal_meaning(u), C_L)) == 0:
        raise ValueError("L0 is only defined if u is consistent with C_L")
    probs = {}
    for C_S in contexts:
        probs[frozenset(C_S)] = L0_un_norm(C_S, u, C_L)
    return normalize_probs(probs)

L0_cache = {}
"""Usage: L0_cache[(u, C_L)][C_S]"""
for u in utterances:
    for C_L in contexts:
        try:
            L0_cache[(u, frozenset(C_L))] = L0_norm(u, C_L)
        except ValueError:
            continue


# ============ Speaker =============
# S1(u | C_S) =~ E_C_L' Util(u, C_S, C_L') * P(u)
# Util(u, C_S, C_L) = exp(a * log(L0(C_S | u, C_L) - Cost(u))

def utterance_cost(u):
    return 0 if u is "null" else cost_factor * len(u.split("_"))



# new for probing utilities
def get_utilities(u, C_S):
    """returns function: \C_L[Util(u | C_S, C_L)]"""
    """Util(u, C_S, C_L) = exp(a * (log(L0(C_S | u, C_L)) - Cost(u)))"""
    u_m = literal_meaning(u)
    C_L_prior = listener_context_prior[frozenset(C_S)]
    utilities = {}
    for C_L in C_L_prior.keys():
        try:
            L0 = L0_cache[(u, frozenset(C_L))][frozenset(C_S)]
            cost = utterance_cost(u)
            if L0 == 0:
                utilities[frozenset(C_L)] = 0
            else:
                utilities[frozenset(C_L)] = math.exp(alpha * (math.log(L0) - cost))
        # except ValueError:
        #     continue
        except KeyError:
            continue
    return utilities

utilities_cache = {}
for u in utterances:
    for C_S in contexts:
        utils = get_utilities(u, C_S)
        for C_L in contexts:
            try:
                utilities_cache[(u, frozenset(C_S), frozenset(C_L))] = utils[frozenset(C_L)]
            except KeyError:
                continue


def get_expected_utility(u, C_S):
    cumulative_utility = 0
    for C_L in contexts:
        try:
            C_L_prob = listener_context_prior[frozenset(C_S)][frozenset(C_L)]
            cumulative_utility += utilities_cache[(u, frozenset(C_S), frozenset(C_L))] * C_L_prob
        except KeyError:
            continue
    return cumulative_utility


expected_utilities_cache = {}
for C_S in contexts:
    for u in utterances:
        try:
            expected_utilities_cache[(u, frozenset(C_S))] = get_expected_utility(u, C_S)
        except KeyError:
            continue


def S1_un_norm(u, C_S):
    # S1(u | C_S) =~ E_C_L' Util(u, C_S, C_L') * P(u)
    # Util(u, C_S, C_L) = exp(a * log(L0(C_S | u, C_L) - Cost(u))
    return expected_utilities_cache[(u, frozenset(C_S))]

# # original for archival purposes
# def S1_un_norm(u, C_S):
#     u_m = literal_meaning(u)
#     C_L_prior = listener_context_prior[frozenset(C_S)]
#     utilities = {}
#     for C_L in C_L_prior.keys():
#         L0 = L0_norm(u, C_L)[frozenset(C_S)]
#         cost = utterance_cost(u)
#         utilities[frozenset(C_L)] = math.exp(alpha * (math.log(L0) - cost)) * C_L_prior[frozenset(C_L)]
#     return sum(utilities.values())

def S1_norm(C_S):
    probs = {}
    for u in utterances:
        try:
            probs[u] = S1_un_norm(u, C_S)
        except ValueError:
            pass
    try:
        normalize_probs(probs)
    except ZeroDivisionError:
        pass
    return normalize_probs(probs)

def make_S1_probs():
    S1_probs = {}
    for C_S in contexts:
        try:
            S1_probs[frozenset(C_S)] = S1_norm(C_S)
        except ValueError:
            continue
    return S1_probs

S1_cache = make_S1_probs()
"""Usage: S1(u | C_S) =  S1_cache[C_S][u])"""

# ============ Pragmatic Listener =============
# L1(C_S | u, C_L) =~ S(u | C_S) * P(C_S | C_L)

# P(C_S | C_L) = P(C_L | C_S) * P(C_S) / P(C_L)
#              =o P(C_S)



def L1_un_norm(C_S, u, C_L):
    return S1_cache[frozenset(C_S)][u] * speaker_context_prior[frozenset(C_L)][frozenset(C_S)]


def L1_norm(u, C_L):
    C_Ss = filter(lambda C_S: entails(C_S, C_L) and not entails(C_L, C_S), contexts)
    probs = {}
    for C_S in C_Ss:
        try:
            probs[frozenset(C_S)] = L1_un_norm(C_S, u, C_L)
        except KeyError:
            probs[frozenset(C_S)] = 0
    return normalize_probs(probs)


def make_L1_cache():
    L1_cache = {}
    for u in utterances:
        for C_L in contexts:
            try:
                L1_cache[(u, frozenset(C_L))] = L1_norm(u, C_L)
            except ZeroDivisionError:
                continue
    return L1_cache

L1_cache = make_L1_cache()


# ============= TESTS ===============
my_contexts = [{(1,1)}, {(0,1)}, {(0,0)}, {(1,1), (0,1)}, {(1,1), (0,0)}, {(0,1), (0,0)}, {(1,1), (0,1), (0,0)}, {(1,1)}]
context_names = ["always in Valhalla", "rose to Valhalla", "never in Valhalla", "now in Valhalla", "not change", "not in Valhalla in past", "not fall"]
vals = []
for c in my_contexts:
    # print("L1( C_S=%s | u=\"not_stop\", C_L=W) = " % str(c),
    #       L1_cache[("not_stop", frozenset(literal_meaning("not_present")))][frozenset(c)])
    try:
        vals += [L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset(c)]]
    except KeyError:
        vals += [0]

y_pos = np.arange(len(my_contexts))
plt.bar(y_pos, vals, align='center', alpha=0.5)
plt.xticks(y_pos, context_names, rotation='vertical')
plt.ylabel('L1( C_S=___ | u=\"not_stop\", C_L=now in Hell)')
plt.title('Pragmatic Listener "Satan" \nalpha=6, listener prior={1, 0}, speaker prior={1, 0.5}')
plt.show()

#
# print("L1( C_S={(1,1)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(1,1)})])
# print("L1( C_S={(0,1)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(0,1)})])
# print("L1( C_S={(0,0)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(0,0)})])
# print("L1( C_S={(1,1), (0,1)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(1,1), (0,1)})])
# print("L1( C_S={(1,1), (0,0)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(1,1), (0,0)})])
# print("L1( C_S={(0,1), (0,0)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(0,1), (0,0)})])
# print("L1( C_S={(1,1), (0,1), (0,0)} | u=\"not_stop\", C_L=W) = ", L1_cache[("not_stop", frozenset(literal_meaning("null")))][frozenset({(1,1), (0,1), (0,0)})])

pass