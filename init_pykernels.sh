#! /bin/bash
conda activate cv_env
conda install -c anaconda ipykernel
conda install -c anaconda ipywidgets
python -m ipykernel install --user --name cv_env --display-name "Python (cv_env)"