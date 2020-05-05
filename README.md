Sepsis Framework
==============================
A framework for dealing with the physionet 2019 sepsis dataset and making predictions to be scored against the pre-defined utility function. 

My suggestion is to setup a virtual environment folder at /env/ in the root directory of this project. Install the requirements with 
```
pip install -r requirements.txt
```

Create a `data/` folder in the project directory, make this a symlink if you wish to store it somewhere else, then create a `data/raw` folder. 
1. Run `src/data/download.py` to download the raw data into `data/raw`.
2. Run `src/data/make_frame.py` to convert the data into a dataframe format useful for visualisation in notebooks. 
3. Run `src/data/preprocess.py` to perform various preprocessing steps (that one may want to change if running your own models) that includes simple feature derivation, and converts the ragged data into a nan filled tensor.
4. Run `src/models/predict_model.py` to run a simple model and get cross-validated scores. 
