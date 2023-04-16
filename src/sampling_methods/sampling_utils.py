import math


REGRESSION = 'regression'
DEFAULT_PCT = [0.05] + [i/10 for i in range(1, 5, 1)] + [0.75]


def sample_from_groups_round_up(grouped, p):
    return grouped.apply(lambda g: g.sample(n=math.ceil(p * len(g)))).sample(frac=1).reset_index(level=0, drop=True)

