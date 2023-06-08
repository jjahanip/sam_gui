@echo off

REM Activate the desired Conda environment
call conda activate sam

REM Reset and clear the output of the Jupyter Notebook file
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace 2_instance_segmentation_with_box_annotator.ipynb

REM Open the main Jupyter Notebook file
jupyter notebook 2_instance_segmentation_with_box_annotator.ipynb
