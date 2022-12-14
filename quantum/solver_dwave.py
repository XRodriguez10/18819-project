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

import dimod
from dwave.system import LeapHybridCQMSampler
import pickle

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

    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        break
    img = img.numpy()

    return model, img


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


def parse_sample(sample):
    perturb = np.zeros(784)
    variables = sample.sample

    for i in range(784):
        perturb[i] = variables[f"x{i+21}"]

    return (sample.energy, perturb)


def main(args):
    """Main logic"""
    model, img = load_pretrained_model()

    TIME_LIMIT = 5

    logger.info("")
    logger.info(f"Solving the final LP with D-Wave...")

    with open('model.lp', 'rb') as f:
        cqm = dimod.lp.load(f)

        if args.sampleset:
            with open('sampleset', 'wb') as sampleset_file:
                sampler = LeapHybridCQMSampler()
                sampleset = sampler.sample_cqm(
                    cqm,
                    time_limit=TIME_LIMIT,
                    label="18819 - Barebones code for solving CQM",
                )
                sampleset_file.write(pickle.dumps(sampleset.to_serializable(use_bytes=True)))

        logger.info("...done")
        unpickle_file = open('sampleset', 'rb')
        sampleset = dimod.SampleSet.from_serializable(pickle.load(unpickle_file))

        logger.info("{} feasible solutions of {}.".format(sampleset.record.is_feasible.sum(),
                                                          len(sampleset)))
        # logger.info(f"Best sample: {sampleset.filter(lambda row: row.is_feasible).first}")

        perturb = parse_sample(sampleset.filter(lambda row: row.is_feasible).first)

        # Compare the prediction result of the original and perturbed image
        img_new = img + perturb
        img_new = torch.from_numpy(img_new).float().cpu()
        original_pred = model(torch.from_numpy(img).float().cpu())
        perturbed_pred = model(img_new)
        original_class = torch.argmax(original_pred)
        perturbed_class = torch.argmax(perturbed_pred)

        logger.info("")
        logger.info(
            f"Probability distribution over 10 classes given the original image:\n{original_pred.detach().numpy()}"
        )
        logger.info(
            f"Probability distribution over 10 classes given the perturbed image:\n{perturbed_pred.detach().numpy()}"
        )
        logger.info(f"Prediction given the original image: class {original_class}")
        logger.info(f"Prediction given the worst perturbed image: class {perturbed_class}")
        if original_class == perturbed_class:
            logger.info(f"No adversarial input exists for this model.")
        else:
            logger.info(f"There exists adversarial inputs for this model.")


def get_input_args():
    """Create and parse arguments."""
    parser = argparse.ArgumentParser(
        description="Quantum solver to the adversarial attack problem.")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug messages",
    )
    parser.add_argument(
        "-s",
        "--sampleset",
        action="store_true",
        help="generate new sampleset",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_input_args()
    logger = setup_logger("quantum", verbose=args.verbose)
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
        logger.info("Successfully ran the D-Wave hybrid CQM solver.")
