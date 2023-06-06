@echo off

REM Activate the desired Conda environment
call conda activate sam

REM Reset and clear the output of the Jupyter Notebook file
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace sam_segmenter.ipynb

REM Open the main Jupyter Notebook file
jupyter notebook sam_segmenter.ipynb
