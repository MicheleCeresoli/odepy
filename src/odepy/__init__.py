"""
odepy

An Extremely Fast Propagator for Python, in Python
"""

__title__ = "odepy"
__copyright__ = "© 2022 Michele Ceresoli, Andrea Pasquale"

from .version import __version__

from .integrator import fast_ivp, fast_ivpe, solve
from .events import event_signature, assemble_events
from .tableaus import Vern7, Vern8, Vern9, DOP853, BS5
from .controllers import integral_controller, PI33, PI34, PI42, H211PI, H312PID
from .utils import fun_signature
