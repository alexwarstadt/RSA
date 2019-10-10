from presupposition.set_operations import *

def make_worlds(n_people):
    def make_worlds_helper(n, worlds_so_far):
        new_worlds = []
        if n == 0:
            return worlds_so_far
        else:
            for w in worlds_so_far:
                new_worlds.append(w + [1])
                new_worlds.append(w + [0])
            return make_worlds_helper(n-1, new_worlds)
    return set([tuple(x) for x in make_worlds_helper(n_people, [[]])])


def make_utterances(n_people):
    return ["every", "some", "none", "null"] + \
           ["every_X" + str(i) for i in range(n_people)] + \
           ["some_X" + str(i) for i in range(n_people)]


def make_literal_meaning(worlds, n_people):
    def literal_meaning(u):
        if u == "every":
            return {tuple([1 for i in range(n_people)])}
        if u == "some":
            return worlds.difference({tuple([0 for i in range(n_people)])})
        if u == "none":
            return {tuple([0 for i in range(n_people)])}
        if u == "null":
            return worlds
        if u.startswith("every_X"):
            add_world = [1 for i in range(n_people)]
            add_world[int(u.split("X")[-1])] = 0
            return literal_meaning("every").union({tuple(add_world)})
        if u.startswith("some"):
            subtract_world = [0 for i in range(n_people)]
            subtract_world[int(u.split("X")[-1])] = 1
            return literal_meaning("some").difference({tuple(subtract_world)})
        if u.startswith("observe_n"):
            return set(filter(lambda w: w[int(u.split("n")[-1])] == 0, worlds))
        elif u.startswith("observe_"):
            return set(filter(lambda w: w[int(u.split("_")[-1])] == 1, worlds))
    return literal_meaning


n_people = 3
worlds = make_worlds(n_people)
contexts = powerset(worlds)
utterances = make_utterances(n_people)
literal_meaning = make_literal_meaning(worlds, n_people)

observe_0 = literal_meaning("observe_0")
observe_1 = literal_meaning("observe_1")
observe_2 = literal_meaning("observe_2")
observe_n0 = literal_meaning("observe_n0")
observe_n1 = literal_meaning("observe_n1")
observe_n2 = literal_meaning("observe_n2")