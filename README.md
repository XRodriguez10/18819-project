# Evaluate Neural Network Robustness against Adversarial Attacks

CMU 18-819, Fall 2022

- Claudio Gomes (cpratago@andrew.cmu.edu)
- Ke Xu ()
- Yuxuan Zheng (yzheng5@andrew.cmu.edu)

---

This is the repository to hold the artifacts of our project to evaluate the robustness of a pre-trained neural network with both the classical and quantum approaches.

Structure of the files:

```
Project Repository
│   notebook.ipynb              # Notebook to explain the project and execute the solvers
│   model.lp                    # MILP model dumped by CPLEX
│   sampleset                   # ???
│
└─── classical
│   │   solver_mip.py           # The classical solver
│   │   xu_net.py               # Neural network structure
│   │   xu_net_for_MNIST.pth    # Pre-trained model parameters
│
└─── quantum
│   │   solver_dwave.py         # The quantum solver
│
└─── utils
│   │   common.py               # Helper functions
│
│
```
