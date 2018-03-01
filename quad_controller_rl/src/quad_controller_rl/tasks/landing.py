"""Landing task."""

import numpy as np
import math
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    """Lift off the ground, reach a target height and maintain a position"""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        # print("Landing(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        # print("Landing(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 30.0  # secs
        self.max_error_position = 8.0  # distance units
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to hover at
        self.weight_position = 0.5
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.3
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.2
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to land at
        self.target_z = 0.0  # target height (z position) to reach for successful landing

    def reset(self):
        print('resetting')
        self.last_timestamp = None
        self.last_position = None
        p = self.target_position + np.random.normal(1.0, 0.5, size=3)  # slight random position around the target
        return Pose(
            position=Point(*p),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        # print("Landing(): velocity = {}".format(velocity))  # [debug]
        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done, reward = self.compute_reward_with_error(False, pose, state, timestamp, position)
        # done, reward = self.compute_reward(position)
        # done, reward = self.compute_reward(False, pose, timestamp)

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

    # def compute_reward(self, position):
    #     done = False
    #     done = all(position <= self.target_position)
    #     reward = 0
    #     if done:
    #         reward = 100.0
    #     return done, reward

    def compute_reward(self, done, pose, timestamp):
        # reward = zero for matching target z, -ve as you go farther, up to -20
        reward = -min(abs(self.target_position[2] - pose.position.z), 20.0)
        if pose.position.z == self.target_position[2]:  # agent has crossed the target height
            reward += 10.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        return done, reward

    def compute_reward_with_error(self, done, pose, state, timestamp, position):
        error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])  # Euclidean distance from target orientation quaternion (a better comparison may be needed)
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])  # Euclidean distance from target velocity vector

        reward = -(self.weight_position * error_position +
                   self.weight_orientation * error_orientation +
                   self.weight_velocity * error_velocity)

        if pose.position.z == self.target_position[2]:  # agent has crossed the target height
            reward += 10.0  # bonus reward
            done = True
        elif error_position > self.max_error_position:
            reward -= 10.0  # extra penalty, agent strayed too far
            done = True
        elif timestamp > self.max_duration:
            reward += 10.0  # extra reward, agent made it to the end
            done = True
        # elif all(position <= self.target_position):
        #     reward += 100.0  # extra reward, agent made it to the end
        #     done = True
        # # print('reward', reward)
        return done, reward

