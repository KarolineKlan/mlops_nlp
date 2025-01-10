# nlp

This is the final project in the course 02476 Machine Learning Operations for group 42. This repository contains a project that uses the [IMDB dataset](https://pytorch.org/text/stable/datasets.html#imdb) to train a binary sentiment analysis model. The dataset contains 50,000 movie reviews, each labeled to identify whether the review is positive or negative. The dataset is split into 25,000 reviews for training as well as 25,000 for testing, resulting in a 50% split between the training and test sets. 

The overall goal of this project is to classify movie reviews as either positive or negative. The project demonstrates various aspects of Machine Learning Operations (MLOps), including data preprocessing, model training, evaluation, and deployment using Docker.

Initially we will use the pretrained DistilBERT model which is a distilled version of the BERT (Bidirectional Encoder Representations from Transformers) model from the HuggingFace Transformers library. This model will be used to preprocess the data by converting it into embeddings. 
We then expect to use PyTorch lightning to implement a shallow classifier model to classify the embedded text into two categories. 




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
