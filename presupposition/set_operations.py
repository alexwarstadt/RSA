
def is_true_at(proposition, world):
    return world in proposition


def entails(p1, p2):
    return p1.issubset(p2)


def intersect(p1, p2):
    return p1.intersection(p2)


def powerset(the_set):
    if len(the_set) == 1:
        return [the_set]
    else:
        to_return = [the_set]
        for x in the_set:
            sub_powerset = powerset(the_set.difference({x}))
            for s in sub_powerset:
                if s not in to_return:
                    to_return.append(s)
        return to_return


def conditional_probability(A, B):
    return len(intersect(A, B)) / len(B)


def normalize_probs(probs):
    total = sum(probs.values())
    for key in probs.keys():
        probs[key] = probs[key] / total
    return probs


def proposition_probability(proposition, distribution):
    cumulative_probability = 0
    for p in powerset(proposition):
        try:
            cumulative_probability += distribution[frozenset(p)]
        except KeyError:
            continue
    return cumulative_probability



