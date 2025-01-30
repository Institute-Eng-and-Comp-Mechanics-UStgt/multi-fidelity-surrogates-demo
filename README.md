# Multi-fidelity Surrogates Demo
Demo of multi fidelity approaches on a toy problem and an engineering use case.

## Reference
A reference to the paper "_Multi-Fidelity Surrogate Model for Representing Hierarchical and Conflicting Databases to Approximate Human-Seat Interaction_" will be added once it is published.

## Abstract
It has been shown that working with databases from heterogeneous sources of varying fidelity can be leveraged in multi-fidelity surrogate models to enhance the high-fidelity prediction accuracy or, equivalently, to reduce the amount of high-fidelity data and thus computational effort required while maintaining accuracy. In contrast, this contribution leverages low-fidelity data queried on a larger feature space to realize data-driven multi-fidelity surrogate models with a fallback option in regimes where high-fidelity data is unavailable. Accordingly, methodologies are introduced to fulfill this task and effectively resolve the contradictions, that inherently arise in multi-fidelity databases. In particular, the databases considered in this contribution feature two levels of fidelity with a defined hierarchy, where data from a high-fidelity source is, when available, prioritized over low-fidelity data. The proposed surrogate model architectures are illustrated first with a toy problem and further examined in the context of an engineering use case in autonomous driving, where the human-seat interaction is evaluated using a data-driven surrogate model, that is trained through an active learning approach. It is shown, that two proposed architectures achieve an improvement in accuracy on high-fidelity data while simultaneously performing well where high-fidelity data is unavailable compared to a naive approach.

## Contents of the Repository
- `data.mat`: Data for the engineering use case as binary MATLAB® file.
- `mf_on_engineering_problem.ipynb`: Jupyter notebook with the engineering use case.
- `mf_on_toy_problem.ipynb`: Jupyter notebook with the toy problem.
- `requirements.txt`: Required packages to run the notebooks.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.