"""
Solve the MIP problem.
"""

import sys
import argparse
import os

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.common import setup_logger

batch_size = 1


def load_pretrained_model():
    """Load the pre-trained model and fetch a single image from the test dataset."""
    project_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    logger.debug("Project path: {}".format(project_path))

    data_path = os.path.join(project_path, "data")
    model_path = os.path.join(project_path, "classical/xu_net_for_MNIST.pth")

    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load(model_path, map_location=torch.device('cpu'))

    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        break
    img = img.numpy()

    return model, img


def get_model_params(model):
    """Get the parameters of the model."""
    weights_and_bias = []
    weights = []
    bias = []

    for name, param in model.named_parameters():
        weights_and_bias.append(param.cpu().detach().numpy())

    for i in range(len(weights_and_bias)):
        if i % 2 == 0:
            weights.append(weights_and_bias[i])
        else:
            bias.append(weights_and_bias[i].reshape(1, -1))

    logger.info("Num of weights: {}".format(len(weights)))
    logger.info("Num of biases: {}".format(len(bias)))

    return weights, bias


def construct_math_program(model, img, weights, bias):
    """
    Construct the math program to solve.
    """
    perturb = cp.Variable((1, 784))
    constraints = [-0.1 <= perturb, perturb <= 0.1]
    x = img + perturb
    layer_idx = 0
    lb = []
    ub = []
    a = []
    y_list = []

    for name, layer in model.named_modules():
        if type(layer) is nn.Linear:
            logger.info("Considering: name: {}, layer: {}".format(name, layer))
            x = x @ weights[layer_idx].T + bias[layer_idx]
            row, col = x.shape
            lb_temp = np.zeros_like(x, shape=x.shape)
            ub_temp = np.zeros_like(x, shape=x.shape)
            for i in range(row):
                for j in range(col):
                    obj = cp.Minimize(x[i][j])
                    prob = cp.Problem(obj, constraints)
                    lb_temp[i][j] = prob.solve()
                    obj = cp.Maximize(x[i][j])
                    prob = cp.Problem(obj, constraints)
                    ub_temp[i][j] = prob.solve()
            lb.append(lb_temp)
            ub.append(ub_temp)

            logger.info("  lower bound: ")
            print(lb_temp)
            logger.info("  upper bound: ")
            print(ub_temp)

            layer_idx += 1
        elif type(layer) is nn.ReLU:
            logger.info("Considering: name: {}, layer: {}".format(name, layer))
            a_temp = cp.Variable(x.shape, boolean=True)
            row, col = x.shape
            lb_temp = lb[layer_idx - 1]
            ub_temp = ub[layer_idx - 1]
            y = cp.Variable(x.shape)
            for i in range(row):
                for j in range(col):
                    constraints.append(y[i][j] <= x[i][j] - lb_temp[i][j] * (1 - a_temp[i][j]))
                    constraints.append(y[i][j] >= x[i][j])
                    constraints.append(y[i][j] <= ub_temp[i][j] * a_temp[i][j])
                    constraints.append(y[i][j] >= 0)
            x = y
            y_list.append(y)
            a.append(a_temp)

    # Define the objective function
    obj = cp.Minimize(x[0][7] - x[0][2])
    prob = cp.Problem(obj, constraints)

    return prob, perturb


def main(args):
    """Main logic"""
    model, img = load_pretrained_model()
    weights, bias = get_model_params(model)

    prob, perturb = construct_math_program(model, img, weights, bias)

    optimal_value = prob.solve()
    logger.info("Optimal value reported by the classical solver: {}".format(optimal_value))

    # Compare the prediction result of the original and perturbed image
    img_new = img + perturb.value
    img_new = torch.from_numpy(img_new).float().cpu()
    original_pred = model(torch.from_numpy(img).float().cpu())
    perturbed_pred = model(img_new)

    logger.info("Prediction given the original image:")
    print(original_pred)
    logger.info("Prediction given the perturbed image:")
    print(perturbed_pred)


def get_input_args():
    """Create and parse arguments."""
    parser = argparse.ArgumentParser(
        description="Classical solver to the adversial attack problem.")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug messages",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_input_args()
    logger = setup_logger("main", verbose=args.verbose)
    logger.debug("Received input argument: {}".format(args))

    try:
        main(args)
    except Exception as e:
        logger.exception("Failed to run the script due to exception: {}".format(e))
    except KeyboardInterrupt as e:
        logger.exception("Exit due to keyboard interrupt: {}".format(e))
    except SystemExit as e:
        if e.code:
            logger.exception("Exit with a non-zero code: {}".format(e))
    except BaseException as e:
        logger.exception("Failed to run the script due to unknown issue: {}".format(e))
    else:
        logger.info("Successfully ran the classical solver.")
