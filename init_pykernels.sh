#! /bin/bash
conda activate nlp_env
conda install -c anaconda ipykernel
conda install -c anaconda ipywidgets
python -m ipykernel install --user --name nlp_env --display-name "Python (nlp_env)"