from definitions import *
from src.features.signatures.compute import RollingSignature


dataset = load_pickle(DATA_DIR + '/interim/from_raw/dataset.dill', use_dill=True)

signatures = RollingSignature(depth=2, logsig=True).compute(dataset.)

