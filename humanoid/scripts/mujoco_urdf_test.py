#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco model="legs">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="/home/lenovo/projects/cheetah_RL/IUST-Cheetah-Software-RL/RL/resource/meshes/" />
    <option integrator="RK4" timestep="0.01"/>
    <size njmax="500" nconmax="100" />
        <default>
            <joint armature="0.0" damping="0.0" limited="false"/>
            <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
        </default>
    <asset>
        <mesh name="trunk" file="trunk.stl" />
        <mesh name="hip" file="hip.stl" />
        <mesh name="thigh_mirror" file="thigh_mirror.stl" />
        <mesh name="calf" file="calf.stl" />
        <mesh name="thigh" file="thigh.stl" />

        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>

    </asset>
    <worldbody>

        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <!-- <geom name="ground" type="plane" conaffinity="1" pos="98 0 0" size="100 .8 .5" material="grid"/> -->
        <geom conaffinity="1" group="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
        

        <body name="base" pos="0.0 0.0 0.5">
            <joint type="free" name="floating_base"/>
            <geom type="mesh" mesh="trunk" group="1"/>

            <inertial pos="-0.0033599 -0.00084664 0.1336" mass="9.9576" diaginertia="0.089966 0.060002 0.056135" />
            <!-- <joint name="floating_base" type='free' limited='false'/> -->

            <site name='imu' size='0.01' pos='0.0 0 0.0'/>
            <body name="FR_hip" pos="0.13 -0.05 0">
                <inertial pos="-0.00266413 -0.0163358 2.49436e-05" quat="0.475134 0.521822 -0.477036 0.523818" mass="0.864993" diaginertia="0.00151807 0.00143717 0.000946744" />
                <joint name="FR_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 1 0 0" type="mesh" group="1" mesh="hip" />

                <body name="FR_thigh" pos="0.06 -0.015 0">
                    <inertial pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="FR_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh_mirror" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->

                    <body name="FR_calf" pos="0 -0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="FR_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0.018 -0.22" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="FL_hip" pos="0.13 0.05 0">
                <inertial pos="0.0227338 0.0102794 2.49436e-05" quat="0.415693 0.415059 0.572494 0.571993" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="FL_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="0.13 0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="FL_thigh" pos="0.06 0.015 0">
                    <inertial pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="FL_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="FL_calf" pos="0 0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="FL_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="RR_hip" pos="-0.13 -0.05 0">
                <inertial pos="-0.0227338 -0.0102794 2.49436e-05" quat="0.415059 0.415693 0.571993 0.572494" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="RR_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 0 0 -1" type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="-0.13 -0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="RR_thigh" pos="-0.06 -0.015 0">
                    <inertial pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="RR_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh_mirror" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="RR_calf" pos="0 -0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="RR_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="RL_hip" pos="-0.13 0.05 0">
                <inertial pos="-0.0227338 0.0102794 2.49436e-05" quat="0.572494 0.571993 0.415693 0.415059" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="RL_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 0 1 0" type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="-0.13 0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="RL_thigh" pos="-0.06 0.015 0">
                    <inertial pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="RL_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="RL_calf" pos="0 0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="RL_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
    </body>
    </worldbody>

    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_calf_joint" gear="150"/>  
    </actuator> 
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    # sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    # sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break