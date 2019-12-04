"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[0])

if ROOT_DIR == '/home/morrill/Documents/sepsis_framework':
    DATA_DIR = '/scratch/morrill/sepsis_framework/data'
else:
    DATA_DIR = ROOT_DIR + '/data'
MODELS_DIR = ROOT_DIR + '/models'

# Packages/functions used everywhere
from src.omni.decorators import *
from src.omni.functions import *
from src.omni.base import *
