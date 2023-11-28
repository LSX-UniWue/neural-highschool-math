# neural-highschool-math
Material for our IEEE ICDM 2023 paper "Can Neural Networks Distinguish High-school Level Mathematical Concepts?"

>Processing symbolic mathematics algorithmically is an important field of research.
>It has applications in computer algebra systems and supports researchers as well as applied mathematicians in their daily work.
>Recently, exploring the ability of neural networks to grasp mathematical concepts has received special attention.
>One complex task for neural networks is to understand the relation of two mathematical expressions to each other.
>Despite the advances in learning mathematical relationships, previous studies are limited by small-scale datasets, relatively simple formula construction by few axiomatic rules and even artifacts in the data.
>With this work, we aim at overcoming these limitations and provide a deeper insight into the representation power of neural networks for classifying mathematical relations.
>We introduce a novel data generation algorithm to allow for more complex formula compositions and fully include mathematical fields up to high-school level.
>We research several tree-based and sequential neural architectures for classifying mathematical relations and conduct a systematic analysis of the models against rule-based as well as neural baselines with a focus on varying dataset complexity, generalization abilities, and understanding of syntactical patterns.
>Our findings show the potential of deep learning models to distinguish high-school level mathematical concepts.

## Data Generator
The code of the dataset generator can be found in the folder `generator`. Install the dependencies given in `requirements.txt`, then run the code as follows: `python3 eqGen.py $n_equiv $n_trans $axioms_path` where `$n_equiv` is the number of axiom substitions, `$n_trans` is the number of transformation steps and `$axioms_path` is the path of the `axioms.txt` file, e.g. `python3 eqGen.py 500 5000 trig_axioms`. We generally set `n_trans = 10*n_equiv`.

To create the syntax dataset, run `python3 syntax_dataset.py equations.json`, where `equations.json` has been generated using the script `eqGen.py`.


## Citation
    Coming soon.
