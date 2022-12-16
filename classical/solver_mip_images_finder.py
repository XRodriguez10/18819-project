"""
Solve the MIP problem.
"""

import sys
import argparse
import os

project_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

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
    logger.info(f"Loading the pre-trained neural network model...")
    logger.debug("Project path: {}".format(project_path))

    data_path = os.path.join(project_path, "data")
    model_path = os.path.join(project_path, "classical/xu_net_for_MNIST.pth")

    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load(model_path, map_location=torch.device('cpu'))

    return model, test_loader


def get_model_params(model):
    """Get the parameters of the model."""
    logger.info(f"Loading the model parameters...")
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

    logger.debug("Num of weights: {}".format(len(weights)))
    logger.debug("Num of biases: {}".format(len(bias)))

    return weights, bias


def construct_math_program(model, img, weights, bias, true, target):
    """
    Construct the math program to solve.
    """
    logger.info(f"Constructing the math program to solve...")
    perturb = cp.Variable((1, 784))
    constraints = [-0.1 <= perturb, perturb <= 0.1]
    x = img + perturb
    layer_idx = 0
    lb = []
    ub = []
    a = []
    y_list = []

    logger.info("")
    logger.info(
        f"Stage 1: solve lower/upper bounds of Linear layers and append constraints for ReLU layers..."
    )

    for name, layer in model.named_modules():
        if type(layer) is nn.Linear:
            # If we hit a Linear layer, we need to solve an LP to get the lower and upper bound of the activations of this layer.
            # The lower and upper bound are used in the next ReLU layer to construct constraints for the final LP problem.
            logger.info("Considering: {}, layer: {}".format(name, layer))
            x = x @ weights[layer_idx].T + bias[layer_idx]
            row, col = x.shape
            lb_temp = np.zeros_like(x, shape=x.shape)
            ub_temp = np.zeros_like(x, shape=x.shape)
            for i in range(row):
                for j in range(col):
                    obj = cp.Minimize(x[i][j])
                    prob = cp.Problem(obj, constraints)
                    lb_temp[i][j] = prob.solve(solver=cp.CPLEX)
                    obj = cp.Maximize(x[i][j])
                    prob = cp.Problem(obj, constraints)
                    ub_temp[i][j] = prob.solve(solver=cp.CPLEX)
            lb.append(lb_temp)
            ub.append(ub_temp)

            layer_idx += 1
        elif type(layer) is nn.ReLU:
            # If we hit a ReLU layer, we want to append multiple constraints (related to the lower and upper bound of the previous Linear layer)
            # to the constraints set.
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

    # Define the final LP problem to solve
    logger.info("")
    logger.info(f"Stage 2: constructing the final LP to solve...")

    # For this particular image, the correct predication is class true
    # Here, we are trying to minimize the predication probability between class true and class target,
    # meaning that we are trying to make the model to make a very wrong prediction towards class target.
    # And we want to see if we can successfully fool the model to change the predication from class true to class target.
    obj = cp.Minimize(x[0][true] - x[0][target])
    prob = cp.Problem(obj, constraints)

    logger.info(f"...done")
    return prob, perturb


def main(args):
    """Main logic"""
    model, test_loader = load_pretrained_model()

    images_found = []

    i = 0
    for data in test_loader:
        img, label = data
        ### Only do it for images of the number 3.
        if label == 3:
            img = img.view(img.size(0), -1)
            img = img.numpy()

            weights, bias = get_model_params(model)

            prob, perturb = construct_math_program(model, img, weights, bias, label, 8)

            logger.info("")
            logger.info(f"Solving the final LP...")
            optimal_value = prob.solve(
                solver=cp.CPLEX,
                cplex_filename=os.path.join(project_path, f"lp_files/image{i}.lp"),
            )

            logger.info("Optimal value reported by the classical solver: {}".format(optimal_value))

            # Compare the prediction result of the original and perturbed image
            img_new = img + perturb.value
            img_new = torch.from_numpy(img_new).float().cpu()
            original_pred = model(torch.from_numpy(img).float().cpu())
            perturbed_pred = model(img_new)
            original_class = torch.argmax(original_pred)
            perturbed_class = torch.argmax(perturbed_pred)

            if original_class == perturbed_class:
                logger.info(f"No adversarial input exists for this model for image{i}.")
            else:
                logger.info(f"There exists adversarial inputs for this model for image{i}.")
                images_found.append(i)
            
            logger.info(f"Found so far: {images_found}.")
        i+=1


def get_input_args():
    """Create and parse arguments."""
    parser = argparse.ArgumentParser(
        description="Classical solver to the adversarial attack problem.")

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
    logger = setup_logger("classical", verbose=args.verbose)
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
        logger.info("Successfully ran the classical solver using CPLEX.")
