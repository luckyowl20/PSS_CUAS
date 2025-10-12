import pybullet as p
import pybullet_data
from math import pi
import time

# ----------------- sim init -----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)
p.loadURDF("plane.urdf")

q = p.getQuaternionFromEuler

# ----------------- load tube -----------------
Lt = 0.60 # length
rt = 0.05 # radius

tube_id = p.loadURDF("tube.urdf",
                     basePosition=[0, 0, Lt/2],
                     baseOrientation=q([0,0,0]),
                     useFixedBase=False)

ball_cid = p.createConstraint(parentBodyUniqueId=tube_id, parentLinkIndex=-1,
                              childBodyUniqueId=-1, childLinkIndex=-1,
                              jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0],
                              parentFramePosition=[0,0,-Lt/2],  # bottom of tube (local)
                              childFramePosition=[0,0,0])       # world origin


# ----------------- load two actuators -----------------
act_urdf = "linear_actuator.urdf"
slider_link = 0                        
tip_in_slider = [0, 0, 0.26]           # attachment point (local to slider link)

# Put actuators to the left/right of tube, pointing toward the tube (±X).
clearance = 0.12                       # distance from tube surface to actuator base
z_mount = 0.3                         # height where actuators sit (tweak as needed)

# Left actuator: rotate +90° about Y so its +Z becomes world +X
# Left actuator at negative X, pointing +X
left_base = [-(rt + clearance), 0.0, z_mount]   # x ≈ -0.17
left_id = p.loadURDF(act_urdf,
                     basePosition=left_base,
                     baseOrientation=q([0,  pi/2, 0]),   # +Z -> +X
                     useFixedBase=False)

p.createConstraint(left_id, -1, -1, -1,
                   jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0],
                   parentFramePosition=[0,0,0],
                   childFramePosition=left_base)

# Right actuator: rotate -90° about Y so its +Z becomes world -X
right_base = [(rt + clearance), 0.0, z_mount]   # x ≈ +0.17
right_id = p.loadURDF(act_urdf,
                      basePosition=right_base,
                      baseOrientation=q([0, -pi/2, 0]),  # +Z -> -X
                      useFixedBase=False)
p.createConstraint(right_id, -1, -1, -1,
                   jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0],
                   parentFramePosition=[0,0,0],
                   childFramePosition=right_base)

# ----------------- attach actuators to tube -----------------
z_attach = 0.01   # 3 cm down from the very top
left_attach_on_tube  = [-rt, 0, z_attach]   # left side surface (local to tube)
right_attach_on_tube = [rt, 0, z_attach]   # right side surface

# Ball joints from slider tips to those tube points:
tip_in_slider = [0, 0, 0.26]

# Ball joints from slider tips to those tube points:
p.createConstraint(left_id,  slider_link, tube_id, -1,
                   jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0],
                   parentFramePosition=tip_in_slider,
                   childFramePosition=left_attach_on_tube)

p.createConstraint(right_id, slider_link, tube_id, -1,
                   jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0],
                   parentFramePosition=tip_in_slider,
                   childFramePosition=right_attach_on_tube)

# ----------------- drive the actuators -----------------
j = 0  # prismatic joint index

# Give them some initial extension so tips reach the tube comfortably
p.resetJointState(left_id,  j, 0.10)
p.resetJointState(right_id, j, 0.10)

# Simple demo: ping-pong different targets to tilt the tube
target_L, target_R = 0.18, 0.02
dirL, dirR = -1, +1
force = 300

def find_prismatic_joints(body_id):
    """Return list of (jointIndex, jointName) for prismatic joints in this body."""
    out = []
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        jtype = info[2]
        if jtype == p.JOINT_PRISMATIC:
            out.append((j, info[1].decode()))
    return out


left_pris  = find_prismatic_joints(left_id)
right_pris = find_prismatic_joints(right_id)

assert len(left_pris)  == 1,  f"Expected 1 prismatic on left, got {left_pris}"
assert len(right_pris) == 1,  f"Expected 1 prismatic on right, got {right_pris}"

JL = left_pris[0][0]   # left joint index
JR = right_pris[0][0]  # right joint index

L_MIN, L_MAX = 0.0, 0.15

ui_target_L = p.addUserDebugParameter("Left target (m)",  L_MIN, L_MAX, 0.10)
ui_target_R = p.addUserDebugParameter("Right target (m)", L_MIN, L_MAX, 0.10)
ui_kp       = p.addUserDebugParameter("Kp (posGain x100)",  0, 500, 120)
ui_kd       = p.addUserDebugParameter("Kd (velGain x10)",   0,  50,   4)
ui_force    = p.addUserDebugParameter("Max Force (N)",      0, 400, 200)

t0 = time.time()
last_print = t0
while p.isConnected():
    # Read sliders
    target_L = p.readUserDebugParameter(ui_target_L)
    target_R = p.readUserDebugParameter(ui_target_R)
    kp = p.readUserDebugParameter(ui_kp) / 100.0
    kd = p.readUserDebugParameter(ui_kd) / 10.0
    fmax = p.readUserDebugParameter(ui_force)

    # Command both actuators (POSITION_CONTROL)
    p.setJointMotorControl2(left_id,  JL, p.POSITION_CONTROL,
                            targetPosition=target_L,
                            positionGain=kp, velocityGain=kd, force=fmax)
    p.setJointMotorControl2(right_id, JR, p.POSITION_CONTROL,
                            targetPosition=target_R,
                            positionGain=kp, velocityGain=kd, force=fmax)

    p.stepSimulation()
    time.sleep(1/240)

    # Read lengths (joint positions) and velocities
    L_pos, L_vel, _, _ = p.getJointState(left_id,  JL)
    R_pos, R_vel, _, _ = p.getJointState(right_id, JR)

    # Print occasionally
    now = time.time()
    if now - last_print > 0.2:
        print(f"left: {L_pos:.3f} m (vel {L_vel:.3f} m/s) | "
              f"right: {R_pos:.3f} m (vel {R_vel:.3f} m/s)")
        last_print = now
