from __future__ import annotations

import pathlib
import re
import traceback
from sys import argv
import copy
import itertools
import json
import logging
import random
from typing import List, Tuple
import numpy as np
import tqdm

import sympy
from mpmath import almosteq, isfinite, mpc, ln, e as E, pi, sin, cos, tan
from wrapt_timeout_decorator import timeout

from equationtree import EquationTree

x, y, z, t = sympy.symbols('x y z t')

operators = frozenset({'Add', 'Mul', 'Pow', 'Sub', 'Div'})  # FIXME
selectable_operators = frozenset(operators - {"Pow", "Div"})
# selectable_operators = frozenset({'Add', 'Mul'})  # , 'Pow'}) # FIXME

# values = frozenset({'x', 'y', 'z', '0', '1', '2', '3', '4'})
values = frozenset({'x', 'y', 'z', '0', '1', '2', '3', '4', '-1', '-2', '-3', '-4'})
subst_values = ['x', 'y', 'z']*10 + ['0', '1', '2', '3', '4', '-1', '-2', '-3', '-4']
constants = ['0', '1', '2', '3', '4', '-1', '-2', '-3', '-4']

variables = {f"var_{i}": i for i in range(3)}


def get_axioms(axioms):
    axioms_dict = dict()
    with sympy.evaluate(False), open(axioms) as f:
        for l in f.readlines():
            if l[0] in {'#', '\n'}:
                continue

            lhs, rhs = l.split('=')
            lhs = EquationTree(str_equation=lhs)
            rhs = EquationTree(str_equation=rhs)
            axioms_dict[lhs] = rhs
            axioms_dict[rhs] = lhs

    return axioms_dict


def write_to_file(path: pathlib.Path, equations: List, labels: List, maxDepth: int):
    assert len(equations) == len(labels)

    data = [[] for _ in range(maxDepth + 1)]
    pickle_dict = dict()
    for i, eq in enumerate(equations):
        depth = eq.depth()
        eq_str = str(eq)
        d = {}
        p = eq.write_out()
        d['equation'] = {'func': p[0][:-1],
                         'vars': p[1][:-1],
                         'nodeNum': p[2][:-1],
                         'depth': p[3][:-1],
                         'numNodes': str(len(eq.nodes)),
                         'variables': eq.variables}
        d['label'] = int(labels[i])
        d['str'] = eq_str
        d['par'] = eq.write_out_paren()
        if i == 0 and eq.depth() > maxDepth:
            data[0].append(d)
        else:
            data[depth].append(d)

        pickle_dict[eq_str] = (eq, int(labels[i]))

    n_samples = [len(data[i]) for i in range(len(data))]
    print(f"{str(path)}: {n_samples}, Σ {sum(n_samples)}")

    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def generate_equivalent_transformations(n_examples: int, allowed_depth: range,
                                        proto_examples: List[Tuple[EquationTree, EquationTree]], no_progress=True,
                                        max_tries=0):
    def substitute_by_match():
        proto_example = copy.deepcopy(random.choice(generated_example_pairs)[random.choice((0, 1))])
        # print("Old: ", proto_example)
        candidates = map(lambda x: (proto_example.contains_subtree(x[0]), x[1]), axioms_dict.items())
        candidates = filter(lambda x: len(x[0]) > 0 and x[1].inorder() != proto_example.inorder(), candidates)
        candidates = tuple(candidates)
        try:
            applied_candidate = random.choice(tuple(candidates))
            applied_transformation = random.choice(applied_candidate[0])
            # hier check ob wir durch null teilen

            new_example = copy.deepcopy(proto_example)
            new_example.transform(applied_candidate[1], applied_transformation.match_dict,
                                  applied_transformation.matches)
            sn = str(new_example)
            if new_example.depth() in allowed_depth and sn not in all_examples and numerically_equal(sn, "0") != -1:
                generated_example_pairs.append((proto_example, new_example))
                all_examples.add(sn)
                generated_samples_str[sn] = new_example
        except IndexError:
            logging.info(f"No rule for {str(proto_example)} found")
        except RuntimeError:
            logging.info(f"Invalid matchdict for {str(new_example)} and {str(applied_candidate[1])}")

    def extend_both_sides():
        val = random.choice(list(values))
        op = random.choice(list(selectable_operators))
        examples = copy.deepcopy(random.choice(generated_example_pairs))
        examples[0].insert_top(op, val)
        examples[1].insert_top(op, val)
        s1 = str(examples[0])
        s2 = str(examples[1])
        if s1 not in all_examples and s2 not in all_examples and numerically_equal(s1, s2) == 1:
            generated_example_pairs.append(examples)
            all_examples.add(s1)
            all_examples.add(s2)
            generated_samples_str[s1] = examples[0]
            generated_samples_str[s2] = examples[1]

    generated_example_pairs = list(proto_examples)
    generated_samples_str = dict()  # map str(equation) to its tree
    all_examples = set()
    init_size = len(generated_example_pairs)

    if not proto_examples:
        raise RuntimeError("Provide a list of proto examples")
        # generated_example_pairs.append((copy.deepcopy(lhs), copy.deepcopy(rhs)))
        # generated_samples_str[str(lhs)] = lhs
        # generated_samples_str[str(rhs)] = rhs

    if not no_progress:
        print("Generate Equivalent Transformations")
    with tqdm.tqdm(total=n_examples+init_size, disable=no_progress) as pbar:
        old_len = 0
        tries = 0
        while len(generated_example_pairs) < n_examples + init_size and (not max_tries or tries < max_tries):
            if random.random() < 0.95 or proto_examples is not None:  # don't extend the sides in case of available prototypes (e.g. diffs)
                substitute_by_match()
            else:
                extend_both_sides()

            new_len = len(generated_example_pairs)
            pbar.update(new_len-old_len)
            old_len = new_len
            tries += 1

    return list(itertools.starmap(lambda x, y: (str(x), str(y)), generated_example_pairs)), generated_samples_str


def generate_negative_example(eq_trees: list[EquationTree]):
    def replace_node(node):
        if node.tag in operators:
            tree.update_node(node.identifier, tag=random.choice(list(selectable_operators - {node.tag})))
        elif node.tag in ("Constant", "Symbol"):
            new_val = random.choice(list(values - {node.data}))
            tree.update_node(node.identifier, data=new_val, tag=EquationTree.get_tag(new_val))
        else:
            raise NotImplementedError()

        return tree

    def shrink_node(node):
        children = tree.children(node.identifier)
        if len(children) > 1:
            tree.remove_node(random.choice(children).identifier)

        tree.link_past_node(node.identifier)

        return tree

    # def grow_node(node):  # todo
    #     n = tree.create_node()

    def change_both_sides(eq1, eq2):
        val1 = random.choice(list(values))
        op1 = random.choice(list(selectable_operators))
        val2 = random.choice(list(values - {val1}))
        op2 = random.choice(list(selectable_operators))

        eq1.insert_top(op1, val1)
        eq2.insert_top(op2, val2)

    tree_idx = random.randint(0, 1)
    operation = random.randint(0, 2)
    if operation == 0:
        tree = copy.deepcopy(eq_trees[tree_idx])
        node = random.choice(list(tree.filter_nodes(lambda x: x.tag != "Function")))
        tree = replace_node(node)
        neg_pair = (tree, eq_trees[1 - tree_idx])
    elif operation == 1:
        try:
            tree = copy.deepcopy(eq_trees[tree_idx])
            node = random.choice(
                list(tree.filter_nodes(lambda x: x.identifier != tree.root and x not in tree.leaves())))
            tree = shrink_node(node)
            neg_pair = (tree, eq_trees[1 - tree_idx])
        except IndexError:
            return None
    elif operation == 2:
        trees = copy.deepcopy(eq_trees)
        change_both_sides(*trees)
        neg_pair = trees

    return neg_pair


def generate_pair_trees(eq_pair, neg_pair, mapping):
    def build_paired_tree(t1: EquationTree, t2: EquationTree):
        t = EquationTree()
        t.create_node(tag="Equality")
        t1 = copy.deepcopy(t1)
        t1.rename_nodes()
        t2 = copy.deepcopy(t2)
        t2.rename_nodes()
        t.paste(t.root, t1)
        t.paste(t.root, t2)

        return t

    pair_trees = [mapping[x] if isinstance(x, str) else x for x in eq_pair]
    pt = build_paired_tree(*pair_trees)
    if not neg_pair:
        return pt

    while True:
        if random.random() < 0.5:
            neg_pair = generate_negative_example(pair_trees)

        if neg_pair is None:
            continue

        s1 = str(neg_pair[0])
        s2 = str(neg_pair[1])
        try:
            s2_diff = sympy.diff(s2, 'x')
            if s2_diff.has(sympy.I, sympy.zoo, sympy.nan, sympy.pi):
                break

            s2_diff_str = str(EquationTree(s2_diff))
            if numerically_equal(s1, s2) == numerically_equal(s1, s2_diff_str) == -1:
                continue
            if numerically_equal(s1, s2) == numerically_equal(s1, s2_diff_str) == has_constant_offset(s1, s2) == 0:
                break
        except Exception:
            print(traceback.print_exc())
            print(s1)
            print(s2)
            continue

    neg = [mapping[x] if isinstance(x, str) else x for x in neg_pair]

    nt = build_paired_tree(*neg)
    return pt, nt


def calculate_derivatives(all_equations, var, merged_examples):
    def filter_equations():
        for eq in all_equations:
            lhs = str(eq.subtree(eq.children(eq.root)[0].identifier))
            rhs = str(eq.subtree(eq.children(eq.root)[1].identifier))
            if numerically_equal(lhs, "0") == 0:
                yield lhs
            if numerically_equal(rhs, "0") == 0:
                yield rhs

    equations = list(filter_equations())
    diffs = list()
    diffs_mapping = dict()
    equivalent_diffs = None

    for eq in equations:
        diff = sympy.diff(eq, var)

        if not diff.has(sympy.I, sympy.zoo, sympy.nan, sympy.pi):
            try:
                diff_tree = EquationTree(parsed_equation=diff)
            except TypeError as e:
                print(e)
                continue

            if equivalent_diffs is None:
                for group in merged_examples.keys():
                    if numerically_equal(str(diff_tree), group) == 1:
                        equivalent_diffs = list(merged_examples[group])
                        break
                else:
                    equivalent_diffs, m = generate_equivalent_transformations(min(len(equations)*2, 100), range(0, 14),
                                                                              [(diff_tree, diff_tree)], max_tries=100)
                    equivalent_diffs = list(filter(lambda x: numerically_equal(x, "0") != -1, itertools.chain.from_iterable(equivalent_diffs)))  # filter out inf and nan
                    diffs_mapping.update(m)
                    diffs_mapping[str(diff_tree)] = diff_tree

            try:
                random_diff = random.choice(list(equivalent_diffs))
                diffs.append((random_diff, eq))
            except IndexError:
                pass

    return diffs, diffs_mapping


def generate_equivalence_classes(n_examples: int, allowed_depth: range, axioms: pathlib.Path, weight_by_depth: bool = False):
    def free_symbols(eq: str):
        if 'x' in eq:
            return True
        elif 'y' in eq:
            return True
        elif 'z' in eq:
            return True
        return False

    def substitute_both_sides():
        with sympy.evaluate(False):
            #sample an axiom to use
            axiom = random.choice(available_axioms)

            #select substitute expressions
            if weight_by_depth: # (gewichtung nach länge der formel)
                x_subst, y_subst, z_subst = copy.deepcopy(random.choices(available_expressions, weights=expression_weights, k=3))
            else:
                x_subst, y_subst, z_subst = copy.deepcopy(random.choices(available_expressions, k=3))

            lhs, rhs = copy.deepcopy(axiom)
            lhs.subs(['x', 'y', 'z'], [x_subst, y_subst, z_subst])
            rhs.subs(['x', 'y', 'z'], [x_subst, y_subst, z_subst])
            if str(rhs) == str(lhs):
                return

            #check infinity
            if numerically_equal(str(rhs), str(lhs)) == -1:
                #print(rhs)
                #print(lhs)
                return

            symbol_perm = ['x', 'y', 'z']
            random.shuffle(symbol_perm)
            lhs.subs(['x', 'y', 'z'], symbol_perm) #for some reason not working :(
            rhs.subs(['x', 'y', 'z'], symbol_perm) #not working for some reason (only works sometimes)

            # check depth of the generated example
            if lhs.depth() <= max(allowed_depth) and lhs.depth() <= max(allowed_depth):
                generated_example_pairs.append((copy.deepcopy(lhs), copy.deepcopy(rhs)))
                generated_samples_str[str(lhs)] = lhs
                generated_samples_str[str(rhs)] = rhs
                if lhs.depth() <= max(allowed_depth)-2:
                    available_expressions.append(copy.deepcopy(lhs))
                    expression_weights.append(1/(lhs.depth()+1))
                if rhs.depth() <= max(allowed_depth)-2:
                    available_expressions.append(copy.deepcopy(rhs))
                    expression_weights.append(1/(rhs.depth()+1))

    generated_example_pairs = list()
    generated_samples_str = dict()  # map str(equation) to its tree
    available_expressions = [EquationTree(str_equation=val) for val in subst_values]
    expression_weights = [1.0 / (expr.depth() + 1) for expr in available_expressions]
    available_axioms = []
    with open(axioms) as f:
        for l in f.readlines():
            if l[0] in {'#', '\n'}:
                continue
            lhs, rhs = l.split('=')

            #add to example pairs
            lhs_tree = EquationTree(str_equation=lhs)
            rhs_tree = EquationTree(str_equation=rhs)
            generated_example_pairs.append((copy.deepcopy(lhs_tree), copy.deepcopy(rhs_tree)))
            generated_samples_str[str(lhs_tree)] = lhs_tree
            generated_samples_str[str(rhs_tree)] = rhs_tree

            if (free_symbols(lhs)) or (free_symbols(rhs)): #check if no free variables in axiom
                available_axioms.append((lhs_tree, rhs_tree))

    print("Generate Equivalence Classes")
    with tqdm.tqdm(total=n_examples) as pbar:
        old_len = 0
        while len(generated_example_pairs) < n_examples:
            substitute_both_sides()
            new_len = len(generated_example_pairs)
            pbar.update(new_len-old_len)
            old_len = new_len

    return generated_example_pairs, generated_samples_str


def numerically_equal(e1: str, e2: str) -> int:
    @timeout(1)
    def _numerically_equal(e1, e2):
        regex = r"(-?\d+)"
        subst = "mpc(\\1.0)"

        free_symbols = ['x', 'y', 'z']
        n_free_symbols = len(free_symbols)

        sampling_points = np.random.default_rng().uniform(1e-5, 2, (10, n_free_symbols))
        e1 = re.sub(regex, subst, e1, 0, re.MULTILINE)
        e2 = re.sub(regex, subst, e2, 0, re.MULTILINE)
        for v in sampling_points:
            s1 = e1
            s2 = e2
            for i, ve1 in enumerate(free_symbols):
                s1 = s1.replace(ve1, f"mpc({v[i]})")
            for i, ve2 in enumerate(free_symbols):
                s2 = s2.replace(ve2, f"mpc({v[i]})")

            try:
                r1 = eval(s1)
                r2 = eval(s2)
                if not (isfinite(r1) and isfinite(r2)):
                    return -1
            except ZeroDivisionError:
                return -1

            if not almosteq(r1, r2):
                return 0

        return 1

    for _ in range(10):
        try:
            return _numerically_equal(e1, e2)
        except (TimeoutError, OverflowError, MemoryError, RecursionError):
            pass

    return -1


def has_constant_offset(e1: str, e2: str) -> bool:
    @timeout(1)
    def _has_constant_offset(e1, e2):
        regex = r"(-?\d+)"
        subst = "mpc(\\1.0)"

        free_symbols = ['x', 'y', 'z']
        n_free_symbols = len(free_symbols)

        sampling_points = np.random.default_rng().uniform(1e-5, 2, (10, n_free_symbols))
        e1 = re.sub(regex, subst, e1, 0, re.MULTILINE)
        e2 = re.sub(regex, subst, e2, 0, re.MULTILINE)
        offset = None
        for v in sampling_points:
            s1 = e1
            s2 = e2
            for i, ve1 in enumerate(free_symbols):
                s1 = s1.replace(ve1, f"mpc({v[i]})")
            for i, ve2 in enumerate(free_symbols):
                s2 = s2.replace(ve2, f"mpc({v[i]})")

            r1 = eval(s1)
            r2 = eval(s2)

            if offset is None:
                if almosteq(r1-r2, 0):  # exclude equal expressions
                    return False
                if almosteq(r1, 0) or almosteq(r2, 0):  # exclude expressions that are equivalent to the derivative
                    return False

                offset = r1-r2

            elif not almosteq(offset, r1-r2):
                return False

        return True

    for _ in range(10):
        try:
            return _has_constant_offset(e1, e2)
        except (TimeoutError, OverflowError, ZeroDivisionError, MemoryError, RecursionError):
            pass

    return False


def permute_variables(tree: EquationTree):
    tree = copy.deepcopy(tree)
    variables = ['x', 'y', 'z']
    variables_subst = list(variables)
    while not all(map(lambda x: x[0] != x[1], zip(variables, variables_subst))):
        random.shuffle(variables_subst)

    lhs = tree.subtree(tree.children(tree.root)[0].identifier)
    rhs = copy.deepcopy(tree.subtree(tree.children(tree.root)[1].identifier))
    rhs.rename_nodes()
    rhs.subs(variables, variables_subst)
    tree.remove_node(tree.children(tree.root)[1].identifier)
    if numerically_equal(str(lhs), str(rhs)):
        return

    tree.paste(tree.root, rhs)

    return tree


if __name__ == "__main__":
    job_id = argv[4] if len(argv) == 5 else 0
    outpath = pathlib.Path(argv[3])
    outpath.mkdir(exist_ok=True)

    axioms_file = "axioms.txt"
    axioms_dict = get_axioms(outpath/axioms_file)
    train_max_depth = 11
    test_max_depth = 14
    train_range = range(0, train_max_depth)
    test_range = range(train_max_depth, test_max_depth+1)
    start_samples, mapping = generate_equivalence_classes(int(argv[1]), range(0, 14), outpath/axioms_file, weight_by_depth=True)
    transformed, mapping_subst = generate_equivalent_transformations(int(argv[2]), range(0, 14), start_samples,
                                                                     no_progress=False)
    mapping.update(mapping_subst)

    print("Merge Classes using Numerical Heuristic")
    merged_examples = dict()
    for group in tqdm.tqdm(transformed):
        group_example = group[0]
        for sample in merged_examples.keys():
            if numerically_equal(group_example, sample):
                merged_examples[sample] |= set(group)
                break
        else:
            merged_examples[group_example] = set(group)

    pos_examples = list()
    neg_examples = list()
    diff_x_pos_examples = list()
    constant_examples = list()
    permuted_examples = list()

    print("Sampling Groups, Calculating Diffs, and Substitutions and Finding Constant Offsets")
    for group_key, eq_ex in tqdm.tqdm(merged_examples.items()):
        eq_ex = tuple(eq_ex)
        pos_samples = list()
        neg_samples = list()
        c_samples = set()
        perm_samples = list()

        constant_offset_class = set(itertools.chain.from_iterable([list(v) for k, v in merged_examples.items() if has_constant_offset(k, group_key)]))
        not_in_class_samples = set(itertools.chain.from_iterable([list(v) for k, v in merged_examples.items() if k != group_key])) - constant_offset_class
        for i in range(min(len(eq_ex), 500)):  # FIXME
            pos_samples.append(random.sample(eq_ex, k=2))
            perm_samples.append(random.sample(eq_ex, k=2))
            neg_samples.append((random.choice(eq_ex), random.choice(tuple(not_in_class_samples))))
            c_samples_count = 0
            while constant_offset_class and (c_samples_count < 3):
                e1 = random.choice(eq_ex)
                e2 = random.choice(tuple(constant_offset_class))
                if has_constant_offset(e1, e2):
                    c_samples.add((e1, e2))
                    c_samples_count += 1

        trees = [generate_pair_trees(p, n, mapping) for p, n in zip(pos_samples, neg_samples)]
        pt = [p[0] for p in trees]
        nt = [p[1] for p in trees]
        pos_examples.extend(pt)
        neg_examples.extend(nt)
        ct = [generate_pair_trees(c, None, mapping) for c in c_samples]
        constant_examples.extend(ct)

        diffs_x, diff_mapping_x = calculate_derivatives(random.sample(pt, k=int(len(pt)/1.8)),
                                                        sympy.Symbol('x'), merged_examples)
        mapping.update(diff_mapping_x)

        trees_diff_x = [generate_pair_trees(d, None, mapping) for d in diffs_x]
        diff_x_pos_examples.extend([p for p in trees_diff_x])

        perm_trees = [permute_variables(generate_pair_trees(p, None, mapping)) for p in perm_samples]
        permuted_examples.extend(filter(None, perm_trees))

    final_neg_examples = random.sample(neg_examples, k=len(pos_examples))
    if len(constant_examples) > len(pos_examples):
        constant_examples = random.sample(constant_examples, k=len(pos_examples))

    all_examples = pos_examples + final_neg_examples + diff_x_pos_examples + permuted_examples + constant_examples
    labels = [1] * len(pos_examples) + [0] * len(final_neg_examples) + [2] * len(diff_x_pos_examples) + [3] * len(permuted_examples) + [4] * len(constant_examples)
    print(len(pos_examples), len(final_neg_examples), len(diff_x_pos_examples), len(constant_examples), len(permuted_examples))

    train_idx = [1 if x.depth() in range(0, train_max_depth) else 0 for x in all_examples]
    test_idx = [1 if x.depth() in range(train_max_depth, test_max_depth) else 0 for x in all_examples]

    write_to_file(outpath/f"train_ec{argv[1]}_et{argv[2]}_id{job_id}.json", list(itertools.compress(all_examples, train_idx)),
                  list(itertools.compress(labels, train_idx)), maxDepth=train_max_depth)
    write_to_file(outpath/f"test_ec{argv[1]}_et{argv[2]}_id{job_id}.json", list(itertools.compress(all_examples, test_idx)),
                  list(itertools.compress(labels, test_idx)), maxDepth=test_max_depth)

    print("Positive")
    for x in random.sample(pos_examples, k=10):
        print(x)
    print("Negative")
    for x in random.sample(neg_examples, k=10):
        print(x)
    print("Diff X")
    for x in random.sample(diff_x_pos_examples, k=10):
        print(x)
    print("Constant Offset")
    for x in random.sample(constant_examples, k=10):
        print(x)
    print("Subst")
    for x in random.sample(permuted_examples, k=10):
        print(x)
