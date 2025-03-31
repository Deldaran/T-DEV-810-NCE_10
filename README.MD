# Zoidberg 2.0

## Workflow

### Installation

Create a python virtual environment and install the dependencies for the project:

```sh
python3 -m venv venv # tested on python 3.12.8
pip install -r requirements.txt
```

Activate the virtual environment in jupyter notebooks and when using pip/python commands in CLI.

### Install a new dependency

```sh
pip install ipykernel # example
pip freeze > requirements.txt # update dependencies
```