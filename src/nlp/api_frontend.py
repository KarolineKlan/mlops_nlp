import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlops-nlp-cloud/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


if __name__ == "__main__":
    get_backend_url()
