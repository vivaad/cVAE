
# Conditional Variational Autoencoders (cVAE)

**An implementation of Conditional Variational Autoencoders (cVAE) which can be implemented on CPU.**

## Introduction

This repository contains an implementation of Conditional Variational Autoencoders (cVAE). The cVAE is a type of variational autoencoder that conditions the latent space on additional information, making it useful for tasks where additional context is beneficial.

## Original Source

This implementation is based on the original work found in the repository [unnir/cvae](https://github.com/unnir/cvae). We have adapted and extended the code to fit our specific use cases and requirements.

## Repository Structure

- `notebooks/` - Jupyter notebooks containing the implementation and experiments.
- `src/` - Source code for the cVAE model and utility functions.
- `data/` - Directory to store datasets used for training and evaluation.
- `results/` - Directory to store the results of experiments.

## Installation

To set up the environment and install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To train the cVAE model, run the following command in your terminal:

```sh
python cpu_cvae.py 
```

You can find example notebooks in the `notebooks/` directory to get started with training and evaluating the cVAE model.

## Acknowledgments

I would like to thank the authors of the original implementation [unnir/cvae](https://github.com/unnir/cvae) for providing the foundational work that this repository builds upon.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---


