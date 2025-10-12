import pybullet as p
import time
import pybullet_data
from math import pi

# gui physics client
physics_client = p.connect(p.GUI)

# built in pdf path finder
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

# ground plane
plane_id = p.loadURDF("plane.urdf")

# actuator
q = p.getQuaternionFromEuler
actuator1_id = p.loadURDF("linear_actuator.urdf", basePosition=[-2, -2, 1], baseOrientation=q([0, pi/2, 0]), useFixedBase=True)
actuator2_id = p.loadURDF("linear_actuator.urdf", basePosition=[-2, 2, 1], baseOrientation=q([0, pi/2, 0]), useFixedBase=True)


num_joints = p.getNumJoints(actuator1_id)
for j in range(num_joints):
    info = p.getJointInfo(actuator1_id, j)
    print(f"Joint {j}: name={info[1].decode()}, type={info[2]}")

# 7. Add sliders for interactive control
target_slider = p.addUserDebugParameter("Target position (m)", 0.0, 0.15, 0.0)
force_slider = p.addUserDebugParameter("Force (N)", 0, 400, 150)
theta_slider = p.addUserDebugParameter("Theta (deg)", 0, -90, 90)
phi_slider = p.addUserDebugParameter("Phi (deg)", 0, 0, 90)

# 8. Run the simulation loop
joint_index = 0  # first and only joint in this URDF
while p.isConnected():
    target = p.readUserDebugParameter(target_slider)
    max_force = p.readUserDebugParameter(force_slider)

    # Apply position control to the prismatic joint
    p.setJointMotorControl2(
        bodyUniqueId=actuator1_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target,
        force=max_force
    )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)