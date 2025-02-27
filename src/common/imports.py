import os
# from dotenv import load_dotenv
# load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np

from einops import *

# frequent used modules
import gc
import math
import time
import json
import random
import shutil
import logging
import datetime
import argparse
import functools
import itertools
import importlib
import imageio.v3 as imageio
import copy
from copy import deepcopy
from collections import OrderedDict

# clean code & typing
from typing import *
from numbers import Number, Complex, Real, Rational, Integral
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict

# data processing & visualization
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

# debug
import gc
import ipdb
import inspect
import weakref
import traceback
import faulthandler
faulthandler.enable()
from ipdb import launch_ipdb_on_exception

from IPython import embed
from IPython.display import clear_output, display

if ("TELEGRAM_BOT_TOKEN" in os.environ) and ("TELEGRAM_CHAT" in os.environ) and (os.environ.get("https_proxy", "") != ""):
    print("Using tqdm.contrib.telegram")
    from tqdm.contrib.telegram import tqdm, trange
    tqdm = functools.partial(tqdm, dynamic_ncols=True, token=os.environ["TELEGRAM_BOT_TOKEN"], chat_id=os.environ["TELEGRAM_CHAT"], mininterval=10, maxinterval=100)
    trange = functools.partial(trange, dynamic_ncols=True, token=os.environ["TELEGRAM_BOT_TOKEN"], chat_id=os.environ["TELEGRAM_CHAT"], mininterval=10, maxinterval=100)
else:
    from tqdm.auto import tqdm, trange
    tqdm = functools.partial(tqdm, dynamic_ncols=True)
    trange = functools.partial(trange, dynamic_ncols=True)

