# This file is executed when the package is imported.
# It's often used to define package-level variables or to
# make submodules available directly.

# Example:
# You can define variables here that are accessible from any
# module within the package.

# Example:
# package_version = "1.0.0"

# Or, you can make submodules directly accessible:
from .utils import print_duck
from .model import GtNR
from .train_application import TrainingApplication
from .test_application import TestingApplication