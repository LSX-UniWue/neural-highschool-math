import argparse
import itertools
import json
import pathlib
import random

import tqdm

from equationtree import EquationTree
from shuffled_negatives import create_negative
from eqGen import write_to_file, numerically_equal


VOCABULARY = [
    'x', 'y', 'z',
    '0', '1', '2', '3', '4',
    '-1', '-2', '-3', '-4',
    'Add', 'Mul', 'Pow', 'Div', 'Sub',
    'tan', 'cos', 'sin', 'ln',
    'E', 'pi',
    '='
]


def get_token_distribution(eq_str: str):
    eq_str = eq_str.replace('(', '').replace(')', '').strip()
    print(eq_str)
    import numpy as np
    tokens = eq_str.strip().split()
    token_counts = np.array([tokens.count(v) for v in VOCABULARY], dtype=np.float32)
    token_distribution = token_counts / np.sum(token_counts)
    return token_counts, token_distribution


def write_out(equal_class, unrel_class, perm_class, outpath, max_depth):
    smallest = min(len(equal_class), len(unrel_class)) #, len(perm_class))
    unrel_class = random.sample(unrel_class, k=smallest)
    equal_class = random.sample(equal_class, k=smallest)
    write_to_file(outpath, perm_class+equal_class+unrel_class,
                  [3]*len(perm_class)+[1]*len(equal_class)+[0]*len(unrel_class), max_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input_file)
    out_path = in_path.parents[1] / (in_path.parent.name + "_syntax")
    out_path.mkdir(exist_ok=True)
    perm_class = list()
    equal_class = list()
    unrel_class = list()
    max_depth = 0

    print("Opening file: ", in_path)
    with open(in_path) as fi:
        data = json.load(fi)
    equations = list(filter(lambda x: x["label"] == 1, itertools.chain.from_iterable(data)))
    file_counter = 0
    for d in tqdm.tqdm(equations):
        eq = EquationTree(par_equation=d["par"])

        if eq.variables:
            eq.permute_variables()

        neq_eq = create_negative(eq)
        if numerically_equal(str(neq_eq.lhs), str(neq_eq.rhs)):
            continue

        unrel_class.append(neq_eq)
        equal_class.append(eq)

        max_depth = max(max_depth, eq.depth())
        if len(equal_class) >= 1000:
            print("Write Out")
            write_out(equal_class, unrel_class, perm_class, out_path/in_path.with_stem(f"{in_path.stem}-{file_counter}").name, max_depth)
            equal_class.clear()
            unrel_class.clear()
            perm_class.clear()
            file_counter += 1

    write_out(equal_class, unrel_class, perm_class, out_path/in_path.with_stem(f"{in_path.stem}-{file_counter}").name, max_depth)
