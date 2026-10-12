"""pyleap: the Python-side loader library for leap solver output.

One home for everything that knows the on-disk solution formats (the .h5
archive written by leap_solve / io::writeSolutionArchive, and the legacy
multi-CSV+meta.json output), the robots/*.yaml description, and the
barycentric-CGL / Bezier evaluation used to turn node values into
continuous-time references, the archive's optional dense off-node constraint
trace (pyleap.internode), and the capsule/obstacle collision geometry (spheres and cuboids) with its
independent clearance recheck (`pyleap.collision`, imported on demand -- it
needs Pinocchio, and it doubles as a CLI: `python -m pyleap.collision <sol>`).
Consumers: simulation/ and viz/.
"""

from pyleap.internode import InterNodeTraces, TermTrace
from pyleap.reference import ReferenceTrajectory
from pyleap.actuation import InverseDynamicsActuation
from pyleap.robot import FootBox, RobotDescription, load_robot, load_robot_text
from pyleap.solution import ModeData, SolutionData, lam_slices, load_solution

__all__ = [
    "ModeData",
    "SolutionData",
    "load_solution",
    "lam_slices",
    "ReferenceTrajectory",
    "InverseDynamicsActuation",
    "RobotDescription",
    "FootBox",
    "load_robot",
    "load_robot_text",
    "InterNodeTraces",
    "TermTrace",
]
