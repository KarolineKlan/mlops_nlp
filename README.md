# nlp

## Project Description
This is the final project in the course 02476 Machine Learning Operations for group number 42. This repository contains a project that uses the [IMDB dataset](https://pytorch.org/text/stable/datasets.html#imdb) to train a binary sentiment analysis model. The dataset contains 50,000 movie reviews, each labeled to identify whether the review is positive or negative. The dataset is split into 25,000 reviews for training as well as 25,000 for testing, resulting in a 50% split between the training and test sets.

The overall goal of this project is to classify the movie reviews as either positive or negative. Additionally we will be focusing on building a robust and efficient model pipeline to streamline the entire workflow, from data handling to model training and deployment. The project demonstrates various aspects of Machine Learning Operations (MLOps), including data preprocessing, model training, evaluation, and deployment using Docker.

Initially we will use the pretrained DistilBERT model which is a distilled version of the BERT (Bidirectional Encoder Representations from Transformers) model from the HuggingFace Transformers library. This model will be used to preprocess the data by converting it into embeddings.
We then expect to use PyTorch lightning to implement a shallow classifier model to classify the embedded text into two categories.
To ensure reproducibility we will use Hydra to implement config files containing all used hyperparameters data paths and model settings.

We will be using Git for version control throughout the project allowing us to track changes. We will log important training information using weights and biases. For logging general information and debugging, we will utilize Loguru. To ensure code compliance with PEP8 (Python Enhancement Proposal 8) standards, we will use Ruff.  Additionally we will implement comtents from the course as it is introduced to us, like a cloud integration.

## Prerequisites

Make sure you have the following installed on your system:

Conda (either Anaconda or Miniconda)

Python 3.8 or higher (managed via Conda)

## Step-by-Step Instructions

1. Clone the Repository

First, clone the repository to your local machine:

- ```git clone https://github.com/KarolineKlan/mlops_nlp.git```

- ```cd mlops_nlp```

2. Install Invoke

Install invoke using Conda:

- ```conda install -c conda-forge invoke```

You can verify the installation by running:

- ```invoke --version```

3. Create the Environment

This repository includes a custom create-environment function defined in the tasks.py file. Use invoke to create the environment by running:

- ```invoke create-environment```

This function will:

Create a Conda environment with the appropriate name (specified in the script).

4. Install Requirements

Once the environment is set up, install additional Python dependencies using the requirements function. Activate the new environment and run the following command:

- ```conda activate nlp```

- ```invoke requirements```

This function will:

Install dependencies listed in the requirements.txt file (if applicable).

Note that windows users have to manually run the following commands:
- ```pip install -r requirements.txt```
- ```pip install -e .```

### For developers:
There are extra packages you need if you want to make changes to the project. You can install them using the `requirements_dev.txt` by invoking the task `dev_requirements"`:

```invoke dev_requirements```

or installing `requirements_dev.txt` directly with pip:

```pip install -r requirements_dev.txt```

To use pre-commit checks run the following line:
- ```pre-commit install```




## Project structure

The directory structure of the project looks like this:
```txt
├──.dvc                       # .dvc linked to GCP bucket, but not actively in use
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
│   ├── config.yaml           # Wrapper config which is always called, experiment is given as argument
│   ├── cloudbuild.yaml
│   └── experiment/           # Specific config for each experiment
│       ├── exp1.yaml
│       └── exp2.yaml
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       ├── index.md
│       ├── model.md
│       └── train.md
├── models/                   # Temporary folder for trained models, main storage is in a GCP bucket
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_visualize.py
├── .gitignore
├── .pre-commit-config.yaml   # Linting and ruff formatting
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements_dev.txt      # Development requirements
├── requirements_test.txt      # Requirements for unittests
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Viewing Code Documentation Locally
To view the project documentation locally, you can use `mkdocs serve`. This command will start a local web server and automatically rebuild the documentation as you make changes.

### Prerequisites
Ensure you have `mkdocs` and any necessary plugins installed. These are installed in the developer packages (see [For developers](#for-developers))

or alternatively just install the specific packages:
```pip install mkdocs==1.6.1 mkdocs-material==9.4.6 mkdocstrings-python==1.12.2```

### Steps to serve documentation locally
Navigate to the docs directory:

```cd docs```

Serve the documentation as a website on a local server:

```mkdocs serve```

Close the server with `Ctrl+C`.
