import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# from geepee.aep_models import SGPLVM, SGPR, SDGPR
import geepee.aep_models as aep
import geepee.vfe_models as vfe
from geepee.utils import flatten_dict, unflatten_dict
