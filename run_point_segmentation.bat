@echo off

REM Activate the desired Conda environment
call conda activate sam

REM Reset and clear the output of the Jupyter Notebook file
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace 1_semantic_segmentation_with_point_annotator.ipynb

REM Open the main Jupyter Notebook file
jupyter notebook 1_semantic_segmentation_with_point_annotator.ipynb
