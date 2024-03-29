import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, 'input')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')
FASTMMD_MATLAB_PATH = os.path.join(ROOT_DIR, 'test', 'FastMMD')
STRUC2VEC_MAIN_PATH = os.path.join(ROOT_DIR, 'embedding', 'struc2vec', 'src', 'main.py')
