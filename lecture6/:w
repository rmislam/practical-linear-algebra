import pybullet as p
import numpy as np
from IPython import embed

def getTransform(body, linkIndex):
    """Compute the homogeneous transform of the specified link relative to the world transform
    """
    position, orientation = p.getLinkState(body, linkIndex)[4:6]
    transform = np.eye(4)
    transform[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    transform[:3, 3] = position
    return transform

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(1)

robot = p.loadURDF("../bullet3/data/TwoJointRobot_wo_fixedJoints.urdf")  # change path according to your system
#robot = p.loadURDF("../bullet3/data/kuka_lwr/kuka.urdf")  # change path according to your system
#robot = p.loadURDF("../bullet3/data/humanoid/nao.urdf")  # change path according to your system

# open the ipython terminal
embed()
