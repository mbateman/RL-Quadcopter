"""Hover task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Lift off the ground, reach a target height and maintain a position"""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 30.0  # secs
        self.max_error_position = 8.0  # distance units
        self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        # slight random position around the target
        p = self.target_position + np.random.normal(0.5, 0.1, size=3)
        return Pose(
                position=Point(*p),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )
        # return Pose(
        #     position=Point(0.0, 0.0, np.random.normal(10.0, 0.1)),  # drop off from a slight random height
        #     orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
        # ), Twist(
        #     linear=Vector3(0.0, 0.0, 1.0),  # A little linear acceleration to cope up with initial gravity pull
        #     angular=Vector3(0.0, 0.0, 0.0)
        # )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done, reward = self.compute_reward_(pose, timestamp, linear_acceleration)

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

    def compute_reward(self, pose, timestamp):
        # reward = zero for matching target z, -ve as you go farther, up to -20
        done = False
        reward = -min(abs(self.target_position[2] - pose.position.z), 20.0)
        if pose.position.z >= self.target_position[2]:  # agent has crossed the target height
            reward += 10.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        return done, reward

    def compute_reward_(self, pose, timestamp, linear_acceleration):
        done = False
        cur_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        dist = np.linalg.norm(cur_pos - self.target_position)

        if dist <= 3.0:  # agent has to hover around a boundary of 3 points
            reward = 10.0  # reward for each time step the agent hover there
        else:
            reward = -dist - abs(linear_acceleration.z)  # penalize if the agent move beyond the fixed boundary

        if timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        return done, reward