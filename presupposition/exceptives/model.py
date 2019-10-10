from presupposition.exceptives.priors import *
from presupposition.exceptives.constants import *
from presupposition.exceptives.worlds_and_utterances import *
from presupposition.set_operations import *
import math

# ============ Literal Listener =============
# Version 1: L0(C_S | u, C_L) =~ P(u | C_S) * P(C_S | C_L)
# Version 2: L0(C_S | u, C_L) =~ P(C_S | C_L & u)

# def L0_un_norm(C_S, u, C_L):
#     """version 1: L0(C_S | u, C_L) =~ P(u | C_S) * P(C_S | C_L)"""
#     u_m = literal_meaning(u)
#     # if not entails(C_S, C_L):
#     #     return 0
#     if not entails(C_S, u_m):
#         return 0
#     else:
#         return conditional_probability(u_m, C_S) * conditional_probability(C_S, C_L)

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

# def get_expected_utilities(C_S):
#     """returns function: \C_L[Expected-Util(u | C_S)]"""
#     expected_utilities = {}
#     for u in utterances:
#         utilities = get_utilities(u, C_S)
#         cumulative_utility = 0
#         for C_L in contexts:
#             try:
#                 C_L_prob = listener_context_prior[frozenset(C_S)][frozenset(C_L)]
#                 cumulative_utility += utilities[frozenset(C_L)] * C_L_prob
#             except KeyError:
#                 continue
#         expected_utilities[(u, frozenset(C_S))] = cumulative_utility
#     return expected_utilities







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




# L0 TESTS

print("===== L0 Tests =====")

L0_probs = L0_cache[("every", frozenset(literal_meaning("null")))]
print("L0(C_S={(1,1,1)} | u=\"every\", C_L=null) = ", L0_probs[frozenset({(1,1,1)})])
print()

L0_probs = L0_cache[("every_X0", frozenset(literal_meaning("null")))]
print("L0(C_S={(1,1,1)} | u=\"every_X0\", C_L=null) = ", L0_probs[frozenset({(1,1,1)})])
print("L0(C_S={(0,1,1)} | u=\"every_X0\", C_L=null) = ", L0_probs[frozenset({(0,1,1)})])
print("L0(C_S={(1,1,1), (0,1,1)} | u=\"every_X0\", C_L=null) = ", L0_probs[frozenset({(1,1,1), (0,1,1)})])
print()

L0_probs = L0_cache[("every_X0", frozenset(literal_meaning("some")))]
print("L0(C_S={(1,1,1)} | u=\"every_X0\", C_L=some) = ", L0_probs[frozenset({(1,1,1)})])
print("L0(C_S={(0,1,1)} | u=\"every_X0\", C_L=some) = ", L0_probs[frozenset({(0,1,1)})])
print("L0(C_S={(1,1,1), (0,1,1)} | u=\"every_X0\", C_L=some) = ", L0_probs[frozenset({(1,1,1), (0,1,1)})])
print()

L0_probs = L0_cache[("every_X0", frozenset(literal_meaning("every_X1")))]
print("L0(C_S={(1,1,1)} | u=\"every_X0\", C_L=every_X1) = ", L0_probs[frozenset({(1,1,1)})])
print("L0(C_S={(0,1,1)} | u=\"every_X0\", C_L=every_X1) = ", L0_probs[frozenset({(0,1,1)})])
print("L0(C_S={(1,1,1), (0,1,1)} | u=\"every_X0\", C_L=every_X1) = ", L0_probs[frozenset({(1,1,1), (0,1,1)})])
print()

try:
    L0_probs = L0_cache[("every_X0", frozenset(literal_meaning("every")))]
    print("L0(C_S={(1,1,1)} | u=\"every_X0\", C_L=every) = ", L0_probs[frozenset({(1,1,1)})])
    print("L0(C_S={(0,1,1)} | u=\"every_X0\", C_L=every) = ", L0_probs[frozenset({(0,1,1)})])
    print("L0(C_S={(1,1,1), (0,1,1)} | u=\"every_X0\", C_L=every) = ", L0_probs[frozenset({(1,1,1), (0,1,1)})])
    print()
except ZeroDivisionError:
    print("L0(C_S=X | u=\"every_X0\", C_L=every)) is undefined")

try:
    # L0_probs = L0_norm("some", literal_meaning("every"))
    L0_probs = L0_cache[("some", frozenset(literal_meaning("every")))]
    print("L0(C_S={(1,0,0)} | u=\"some\", C_L=every) = ", L0_probs[frozenset({(1,0,0)})])
    print("L0(C_S={(0,1,0)} | u=\"some\", C_L=every) = ", L0_probs[frozenset({(0,1,0)})])
    print("L0(C_S={(0,0,1)} | u=\"some\", C_L=every) = ", L0_probs[frozenset({(0,0,1)})])
    print("L0(C_S={(1,1,1)} | u=\"some\", C_L=every) = ", L0_probs[frozenset({(1,1,1)})])
    print()
except ZeroDivisionError:
    print("L0(C_S=X | u=\"every_X0\", C_L=every)) is undefined")

try:
    # L0_probs = L0_norm("some", literal_meaning("some_X0"))
    L0_probs = L0_cache[("some", frozenset(literal_meaning("some_X0")))]
    print("L0(C_S=[[some]] | u=\"some\", C_L=some_X0) = ", L0_probs[frozenset(literal_meaning("some"))])
    print("L0(C_S=[[some_X0]] | u=\"some\", C_L=some_X0) = ", L0_probs[frozenset(literal_meaning("some_X0"))])
    print("L0(C_S entails 0 came | u=\"some\", C_L=some_X0) = ", proposition_probability(observe_0, L0_probs))
    print("L0(C_S entails 1 came | u=\"some\", C_L=some_X0) = ", proposition_probability(observe_1, L0_probs))
    print("L0(C_S entails 2 came | u=\"some\", C_L=some_X0) = ", proposition_probability(observe_2, L0_probs))
    print()
    L0_probs = L0_cache[("null", frozenset(literal_meaning("some_X0")))]
    print("L0(C_S entails 0 came | u=\"null\", C_L=some_X0) = ", proposition_probability(observe_0, L0_probs))
    print("L0(C_S entails 1 came | u=\"null\", C_L=some_X0) = ", proposition_probability(observe_1, L0_probs))
    print("L0(C_S entails 2 came | u=\"null\", C_L=some_X0) = ", proposition_probability(observe_2, L0_probs))
    print()
except ZeroDivisionError:
    print("L0(C_S=X | u=\"every_X0\", C_L=every)) is undefined")

print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"some\", C_L=[[0]]) = ",
      L0_cache[("some", frozenset(observe_0))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"some\", C_L=[[n0]]) = ",
      L0_cache[("some", frozenset(observe_n0))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"some_X0\", C_L=[[0]]) = ",
      L0_cache[("some_X0", frozenset(observe_0))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"some_X0\", C_L=[[n0]]) = ",
      L0_cache[("some_X0", frozenset(observe_n0))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"null\", C_L=[[0]]) = ",
      L0_cache[("null", frozenset(observe_0))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"null\", C_L=[[n0]]) = ",
      L0_cache[("null", frozenset(observe_n0))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print()

print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"some\", C_L=[[null]]) = ",
      L0_cache[("some", frozenset(literal_meaning("null")))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"some\", C_L=[[null]]) = ",
      L0_cache[("some", frozenset(literal_meaning("null")))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"some_X0\", C_L=[[null]]) = ",
      L0_cache[("some_X0", frozenset(literal_meaning("null")))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"some_X0\", C_L=[[null]]) = ",
      L0_cache[("some_X0", frozenset(literal_meaning("null")))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print("L0(C_S={(1,1,1), (1,1,0), (1,0,1))} | u=\"null\", C_L=[[null]]) = ",
      L0_cache[("null", frozenset(literal_meaning("null")))][frozenset({(1,1,1), (1,1,0), (1,0,1)})])
print("L0(C_S={(0,1,1), (0,1,0), (0,0,1))} | u=\"null\", C_L=[[null]]) = ",
      L0_cache[("null", frozenset(literal_meaning("null")))][frozenset({(0,1,1), (0,1,0), (0,0,1)})])
print()


print("====== Utility ======")

print("U(\"every\", C_S={(1,1,1)}, C_L=[[null]]) = ", utilities_cache[("every", frozenset(literal_meaning("every")), frozenset(literal_meaning("null")))])
print("U(\"every\", C_S=[[some]], C_L=[[null]]) = ", utilities_cache[("every", frozenset(literal_meaning("some")), frozenset(literal_meaning("null")))])
print("U(\"every\", C_S={(1,1,1)}, C_L=[[every]]) = ", utilities_cache[("every", frozenset(literal_meaning("every")), frozenset(literal_meaning("every")))])
try:
    print("U(\"every\", C_S=[[some]], C_L=[[every]]) = ", utilities_cache[("every", frozenset(literal_meaning("some")), frozenset(literal_meaning("every")))])
except KeyError:
    print("U(\"every\", C_S=[[some]], C_L=[[every]]) is undefined")
print("U(\"none\", C_S=[[some]], C_L=[[null]]) = ", utilities_cache[("none", frozenset(literal_meaning("some")), frozenset(literal_meaning("null")))])
print()
#
print("U(\"every_X0\", C_S={(1,1,1)}, C_L=[[null]]) = ", utilities_cache[("every_X0", frozenset(literal_meaning("every")), frozenset(literal_meaning("null")))])
print("U(\"every_X0\", C_S={(1,1,1), (0,1,1)}, C_L=[[null]]) = ", utilities_cache[("every_X0", frozenset(literal_meaning("every_X0")), frozenset(literal_meaning("null")))])
print("U(\"every_X0\", C_S={(0,1,1)}, C_L=[[null]]) = ", utilities_cache[("every_X0", frozenset({(0,1,1)}), frozenset(literal_meaning("null")))])
print("U(\"every_X0\", C_S={(0,1,1)}, C_L=[[not 0]]) = ", utilities_cache[("every_X0", frozenset({(0,1,1)}), frozenset(observe_n0))])
print()

print("U(\"some_X0\", C_S={(1,1,0), (1,0,1)}, C_L=[[null]]) = ", utilities_cache[("some_X0", frozenset({(1,1,0), (1,0,1)}), frozenset(literal_meaning("null")))])
print("U(\"some_X0\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("some_X0", frozenset({(1,1,0), (1,0,1)}), frozenset(observe_0))])
print("U(\"some\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("some", frozenset({(1,1,0), (1,0,1)}), frozenset(observe_0))])
print("U(\"null\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("null", frozenset({(1,1,0), (1,0,1)}), frozenset(observe_0))])
print("U(\"some_X0\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("some_X0", frozenset({(0,1,0), (0,0,1)}), frozenset(observe_n0))])
print("U(\"some\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("some", frozenset({(0,1,0), (0,0,1)}), frozenset(observe_n0))])
print("U(\"null\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("null", frozenset({(0,1,0), (0,0,1)}), frozenset(observe_n0))])
print()

print("U(\"some_X0\", C_S={(1,1,0), (1,0,1)}, C_L=[[null]]) = ", utilities_cache[("some_X0", frozenset({(1,1,0), (1,0,1)}), frozenset(literal_meaning("null")))])
print("U(\"some_X0\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("some_X0", frozenset({(1,1,0), (1,0,1)}), frozenset(literal_meaning("observe_0")))])
print("U(\"some\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("some", frozenset({(1,1,0), (1,0,1)}), frozenset(literal_meaning("observe_0")))])
print("U(\"null\", C_S={(1,1,0), (1,0,1)}, C_L=[[0]]) = ", utilities_cache[("null", frozenset({(1,1,0), (1,0,1)}), frozenset(literal_meaning("observe_0")))])
print("U(\"some_X0\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("some_X0", frozenset({(0,1,0), (0,0,1)}), frozenset(literal_meaning("observe_n0")))])
print("U(\"some\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("some", frozenset({(0,1,0), (0,0,1)}), frozenset(literal_meaning("observe_n0")))])
print("U(\"null\", C_S={(0,1,0), (0,0,1)}, C_L=[[not 0]]) = ", utilities_cache[("null", frozenset({(0,1,0), (0,0,1)}), frozenset(literal_meaning("observe_n0")))])
print()


print("====== Expected Utility ======")
print("EU(\"every\", C_S={(1,1,1)}) = ", expected_utilities_cache[("every", frozenset(literal_meaning("every")))])
print("EU(\"some_X0\", C_S={(1,0,1), (1,1,0), (1,1,1)}) = ", expected_utilities_cache[("some_X0", frozenset({(1,0,1), (1,1,0), (1,1,1)}))])
print("EU(\"some_X0\", C_S={(0,0,1), (0,1,0), (0,1,1)}) = ", expected_utilities_cache[("some_X0", frozenset({(0,0,1), (0,1,0), (0,1,1)}))])
print("EU(\"some\", C_S={(1,0,1), (1,1,0), (1,1,1)}) = ", expected_utilities_cache[("some", frozenset({(1,0,1), (1,1,0), (1,1,1)}))])
print("EU(\"some\", C_S={(0,0,1), (0,1,0), (0,1,1)}) = ", expected_utilities_cache[("some", frozenset({(0,0,1), (0,1,0), (0,1,1)}))])
print()

print("EU(\"some\", C_S=[[some]]) = ", expected_utilities_cache[("some", frozenset(literal_meaning("some")))])
print("EU(\"null\", C_S=[[some]]) = ", expected_utilities_cache[("null", frozenset(literal_meaning("some")))])
print()
# some_x0_prsp_0 = frozenset(intersect(literal_meaning("some_X0"), literal_meaning("observe_0")))
# some_x0_prsp_n0 = frozenset(intersect(literal_meaning("some_X0"), literal_meaning("observe_n0")))
# print("EU(\"every\", C_S=[[every]]) = ", expected_utilities_cache[("every", frozenset(literal_meaning("every")))])
# print("EU(\"some_X0\", C_S=[[some_x0 & 0]]) = ", expected_utilities_cache[("some_X0", some_x0_prsp_0)])
# print("EU(\"some_X0\", C_S=[[some_x0 & n0]]) = ", expected_utilities_cache[("some_X0", some_x0_prsp_n0)])
# print("EU(\"some\", C_S=[[some_x0 & 0]]) = ", expected_utilities_cache[("some", some_x0_prsp_0)])
# print("EU(\"some\", C_S=[[some_x0 & n0]]) = ", expected_utilities_cache[("some", some_x0_prsp_n0)])
# print()
#
#
print("======S1======")
print("S1(\"some\" | C_S={(1,0,1), (1,1,0), (1,1,1)}) = ", S1_cache[frozenset({(1,0,1), (1,1,0), (1,1,1)})]["some"])
print("S1(\"some_X0\" | C_S={(1,0,1), (1,1,0), (1,1,1)}) = ", S1_cache[frozenset({(1,0,1), (1,1,0), (1,1,1)})]["some_X0"])
print("S1(\"some\" | C_S={(0,0,1), (0,1,0), (0,1,1)}) = ", S1_cache[frozenset({(0,0,1), (0,1,0), (0,1,1)})]["some"])
print("S1(\"some_X0\" | C_S={(0,0,1), (0,1,0), (0,1,1)}) = ", S1_cache[frozenset({(0,0,1), (0,1,0), (0,1,1)})]["some_X0"])
print()

print("S1(\"some\" | C_S=[[some]]) = ", S1_cache[frozenset(literal_meaning("some"))]["some"])
print("S1(\"null\" | C_S=[[some]]) = ", S1_cache[frozenset(literal_meaning("some"))]["null"])
print()
#
#
#
print("======L1======")
#
print("L1(C_S=[[every]] | u=\"every\", C_L=[[null]]) = ", L1_cache[("every", frozenset(literal_meaning("null")))][frozenset(literal_meaning("every"))])
print()

print("L1(C_S={(0,1,1)} | u=\"every_X0\", C_L=[[null]]) = ", L1_cache[("every_X0", frozenset(literal_meaning("null")))][frozenset({(0,1,1)})])
print("L1(C_S={(1,1,1)} | u=\"every_X0\", C_L=[[null]]) = ", L1_cache[("every_X0", frozenset(literal_meaning("null")))][frozenset({(1,1,1)})])
print()


print("L1(C_S={(1,0,1), (1,1,0), (1,1,1)} | u=\"some_X0\", C_L=[[0]]) = ",
      L1_cache[("some_X0", frozenset(observe_0))][frozenset({(1,0,1), (1,1,0), (1,1,1)})])
print("L1(C_S={(0,0,1), (0,1,0), (0,1,1)} | u=\"some_X0\", C_L=[[n0]]) = ",
      L1_cache[("some_X0", frozenset(observe_n0))][frozenset({(0,0,1), (0,1,0), (0,1,1)})])
print()

print("L1(C_S={(1,0,1), (1,1,0), (1,1,1)} | u=\"some_X0\", C_L=[[null]]) = ",
      L1_cache[("some_X0", frozenset(literal_meaning("null")))][frozenset({(1,0,1), (1,1,0), (1,1,1)})])
print("L1(C_S={(0,0,1), (0,1,0), (0,1,1)} | u=\"some_X0\", C_L=[[null]]) = ",
      L1_cache[("some_X0", frozenset(literal_meaning("null")))][frozenset({(0,0,1), (0,1,0), (0,1,1)})])
print()


print("L1(C_S entails {(1,0,1), (1,1,0), (1,1,1)} | u=\"some_X0\", C_L=[[0]]) = ",
      proposition_probability(observe_0, L1_cache[("some_X0", frozenset(observe_0))]))
print("L1(C_S entails {(0,0,1), (0,1,0), (0,1,1)} | u=\"some_X0\", C_L=[[n0]]) = ",
      proposition_probability(observe_n0, L1_cache[("some_X0", frozenset(observe_n0))]))
print()

print("L1(C_S entails {(1,0,1), (1,1,0), (1,1,1)} | u=\"some_X0\", C_L=[[null]]) = ",
      proposition_probability(observe_0, L1_cache[("some_X0", frozenset(literal_meaning("null")))]))
print("L1(C_S entails {(0,0,1), (0,1,0), (0,1,1)} | u=\"some_X0\", C_L=[[null]]) = ",
      proposition_probability(observe_n0, L1_cache[("some_X0", frozenset(literal_meaning("null")))]))
print()



pass


# L1_probs = L1_norm(u="some_X0", C_L=literal_meaning("null"))
# print("some_X0")
# print(sum([L1_probs[frozenset(c)] for c in observe_0_contexts]))
#
#
# L1_probs = L1_norm(u="every_X0", C_L=literal_meaning("null"))
# print("every_X0")
# print(L1_probs[frozenset({(0,1,1)})])

