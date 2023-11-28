import itertools
import json
import random
import copy
from equationtree import EquationTree


def create_negative(eq: EquationTree):
    eq = copy.deepcopy(eq)

    # nodes = eq.filter_nodes(lambda x: x.tag != "Equality")
    for child in eq.children(eq.root):
        leaves_vals = list()
        uni_vals = list()
        bin_vals = list()
        for n in eq.subtree(child.identifier).all_nodes_itr():
            n_children = len(eq.children(n.identifier))
            assert 0 <= n_children <= 2
            if n_children == 0:
                leaves_vals.append(copy.copy(n))
            elif n_children == 1:
                uni_vals.append(copy.copy(n))
            elif n_children == 2:
                bin_vals.append(copy.copy(n))

        leaves_vals_shuffled = list(leaves_vals)
        uni_vals_shuffled = list(uni_vals)
        bin_vals_shuffled = list(bin_vals)
        random.shuffle(leaves_vals_shuffled)
        random.shuffle(uni_vals_shuffled)
        random.shuffle(bin_vals_shuffled)

        for old, new in zip(leaves_vals, leaves_vals_shuffled):
            eq.update_node(old.identifier, tag=new.tag, data=new.data)
        for old, new in zip(bin_vals, bin_vals_shuffled):
            eq.update_node(old.identifier, tag=new.tag, data=new.data)
        for old, new in zip(uni_vals, uni_vals_shuffled):
            eq.update_node(old.identifier, tag=new.tag, data=new.data)

    return eq


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    for d in filter(lambda x: x["label"] == 1, itertools.chain.from_iterable(data)):
        eq = EquationTree(par_equation=d["par"])
        print(eq)
        eq = create_negative(eq)
        print(eq)
        print()
