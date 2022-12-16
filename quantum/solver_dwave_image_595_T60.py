"""
Solve the MIP problem.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import cvxpy as cp
import numpy as np
import sys
import argparse
import os


import dimod
from dwave.system import LeapHybridCQMSampler
import pickle


project_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

from utils.common import setup_logger

batch_size = 1


def load_pretrained_model():
    """Load the pre-trained model and fetch a single image from the test dataset."""

    data_path = os.path.join(project_path, "data")
    model_path = os.path.join(project_path, "classical/xu_net_for_MNIST.pth")

    data_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=data_tf)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load(model_path, map_location=torch.device('cpu'))

    return model, test_loader


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
    model, test_loader = load_pretrained_model()

    TIME_LIMIT = 60

    images_found = [595]

    negative_values_found = []
    attacks_found = []

    assert len(images_found) == 1

    i = 0
    for data in test_loader:
        if i in images_found:
            with open(f'lp_files/image{i}.lp', 'rb') as f:
                cqm = dimod.lp.load(f)
            img, label = data
            img = img.view(img.size(0), -1)
            img = img.numpy()

            sampleset_path = f'samplesetsT{TIME_LIMIT}/sampleset{i}'

            run_cqm = False
            if os.path.exists(sampleset_path):
                if os.stat(sampleset_path).st_size == 0:
                    run_cqm = True
            else:
                run_cqm = True

            if run_cqm:
                with open(sampleset_path, 'wb') as sampleset_file:
                    sampler = LeapHybridCQMSampler()
                    print(f'Executing LeapHybridCQMSampler with image{i}')
                    sampleset = sampler.sample_cqm(
                        cqm,
                        time_limit=TIME_LIMIT,
                        label=f"18819 - Checking if adversarial example exists for image {i}",
                    )
                    sampleset_file.write(pickle.dumps(sampleset.to_serializable(use_bytes=True)))
            

            unpickle_file = open(sampleset_path, 'rb')
            sampleset = dimod.SampleSet.from_serializable(pickle.load(unpickle_file))

            (value, perturb) = parse_sample(sampleset.filter(lambda row: row.is_feasible).first)
            
            if value < 0:
                negative_values_found.append(i)

            # Compare the prediction result of the original and perturbed image
            img_new = img + perturb
            img_new = torch.from_numpy(img_new).float().cpu()
            original_pred = model(torch.from_numpy(img).float().cpu())
            perturbed_pred = model(img_new)
            original_class = torch.argmax(original_pred)
            perturbed_class = torch.argmax(perturbed_pred)

            if original_class == perturbed_class:
                logger.info(
                    f"No adversarial input was found for this model for image{i}.")
            else:
                logger.info(
                    f"There exists adversarial inputs for this model for image{i}.")
                attacks_found.append(i)

            logger.info(f"Found {len(attacks_found)} adversarial examples so far: {attacks_found}.")
            print(f"Found {len(attacks_found)} adversarial examples so far: {attacks_found}.")
            logger.info(f"Found {len(negative_values_found)} negative objective values so far: {negative_values_found}.")
            print(f"Found {len(negative_values_found)} negative objective values so far: {negative_values_found}.")
            logger.info(
                f"value:{value}"
            )
            logger.info(
                f"Probability distribution over 10 classes given the original image:\n{original_pred.detach().numpy()}"
            )
            logger.info(
                f"Probability distribution over 10 classes given the perturbed image:\n{perturbed_pred.detach().numpy()}"
            )
        i += 1


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
    logger = setup_logger("quantum", verbose=args.verbose)
    logger.debug("Received input argument: {}".format(args))

    try:
        main(args)
    except Exception as e:
        logger.exception(
            "Failed to run the script due to exception: {}".format(e))
    except KeyboardInterrupt as e:
        logger.exception("Exit due to keyboard interrupt: {}".format(e))
    except SystemExit as e:
        if e.code:
            logger.exception("Exit with a non-zero code: {}".format(e))
    except BaseException as e:
        logger.exception(
            "Failed to run the script due to unknown issue: {}".format(e))
    else:
        logger.info(
            "Successfully ran D-Wave's Hybrid CQM Solver for image 595.")
