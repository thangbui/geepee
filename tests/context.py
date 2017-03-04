import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from geepee.aep_models import SGPLVM
from geepee.utils import flatten_dict, unflatten_dict
