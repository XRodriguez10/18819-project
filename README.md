# Evaluate Neural Network Robustness against Adversarial Attacks

CMU 18-819, Fall 2022

- Claudio Gomes (cpratago@andrew.cmu.edu)
- Ke Xu (kx1@andrew.cmu.edu)
- Yuxuan Zheng (yzheng5@andrew.cmu.edu)

---

This is the repository to hold the artifacts of our project to evaluate the robustness of a pre-trained neural network with both the classical and quantum approach.

Structure of the files:

```
Project Repository
│   notebook.ipynb                    # Notebook to explain the project and execute the solvers
│   model.lp                          # MILP model dumped by CPLEX
│   sampleset                         # Saved sampleset from D-Wave
│
└─── classical
│   │   solver_mip.py                 # The classical solver
│   │   xu_net.py                     # Neural network structure
│   │   xu_net_for_MNIST.pth          # Pre-trained model parameters
|   |   solver_mip_images_finder.py   # Construct .lp files for each image
│
└─── lp_files                         # LP files for evaluating each image
│
└─── quantum
│   │   solver_dwave.py               # The quantum solver
│   │   solver_dwave_image*           # Evaluate a certain image with different sample time
│
└─── samplesetsT*                     # Sample sets with different sample time
│
└─── utils
│   │   common.py                     # Helper functions
│
```
