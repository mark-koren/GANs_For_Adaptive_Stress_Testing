from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.spaces import Discrete_Box
import numpy as np

class CrosswalkEnv(Env):
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        x_p = self._state[3] + action[0]
        y_p = self._state[4] + action[1]
        x_d = x_p - self._state[1]
        y_d = y_p - self._state[2]
        v_old = self._state[0]
        if (x_d >= 5.0) or (y_d/np.maximum(v_old, 1e-7) > 2.0):
            v_new = np.minimum(4.0, v_old + 1.0)
        else:
            v_new = np.maximum(0.0, v_old - 1.0)

        y_c = self._state[2] + 0.5 * (v_old + v_new) * 5.0
        x_c = self._state[1]

        y_d_new = self._state[5] - y_c
        x_d_new = x_d
        if y_d_new < 0.0:
            if x_d_new < 1.0:
                done = True
                reward = 0
            else:
                done = True
                reward = -np.inf
        else:
            done = False
            reward = np.log(1 + self.mahalanobis_d(action))

        self._state = np.array([v_new, x_c, y_c, x_p, y_p])
        observation = np.array([v_new, x_d_new, y_d_new])
        return Step(observation=observation, reward=reward, done=done)

    def mahalanobis_d(self, action):
        #TODO get mean and covariance from G
        #load G
        #bundle = G(self._state)
        #mean = G[0:2].T
        #cov = np.array([[G[2], 0],[0, G[3]])
        mean = np.array([[3.0],[0.0]])
        cov = np.array([[1.5, 0], [0, 2.0]])
        action_v = action.reshape((2,1))

        dif = (action_v - mean)
        inv_cov = np.linalg.inv(cov)
        dist = np.dot(np.dot(dif.T, inv_cov), dif)
        return np.sqrt(dist[0,0])


    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = np.array([4.0, 4.5, 0.0, 6.5, 50.0])
        observation = np.array([self._state[0],
                                self._state[3] - self._state[1],
                                self._state[4] - self._state[2]])
        return observation

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return Box(low=np.array([0.0,0.0]), high=np.array([5.0,2.0]))

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(low=np.array([0.0,0.0,0.0]), high=np.array([4.0, 9.0, 99.0]))

    def render(self):
        print(self._state)