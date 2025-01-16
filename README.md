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

git clone https://github.com/KarolineKlan/mlops_nlp.git
cd mlops_nlp

2. Install Invoke

Install invoke using Conda:

conda install -c conda-forge invoke

You can verify the installation by running:

invoke --version

3. Create the Environment

This repository includes a custom create-environment function defined in the tasks.py file. Use invoke to create the environment by running:

- invoke create-environment

This function will:

Create a Conda environment with the appropriate name (specified in the script).

4. Install Requirements

Once the environment is set up, install additional Python dependencies using the requirements function. Activate the new environment and run the following command:

- conda activate nlp

- invoke requirements

This function will:

Install dependencies listed in the requirements.txt file (if applicable).

Note that windows users have to manually run the following commands:
- pip install -r requirements.txt
- pip install -e .


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
