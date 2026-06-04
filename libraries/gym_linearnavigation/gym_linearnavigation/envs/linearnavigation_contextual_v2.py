import typing as t
from collections import defaultdict
from typing import Any, Dict, Final, List, Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class LinearNavigationContextualV2(gym.Env):
    def __init__(
        self,
        middle_length=60 + 60 + 20,
        blackbox_length=30,
        indicator_steps=1,
        speed='normal',
        n_contexts=2,
        context_override: Optional[int] = None,
        obs_context_override: Optional[int] = None,
        trial_schedule_b_nb_p=('b', 'nb', 'b', 'nb', 'b', 'nb'),
        allowed_contexts: List[int] = None,
        biased_context: bool = False,
        seed=42,
        render_mode='dummy',
    ):

        self.speed_mode: Final[str] = speed

        self.action_space = spaces.Discrete(2)

        self.rz_length: int = 10
        assert blackbox_length == 30
        self.reward_locations_by_context: Dict[int, int] = {
            0: 55,
            1: 65,
            2: 75,
            3: 30 + 70,
            4: 115,
            5: 125,
            6: 135,
            7: 145,
            8: 155,
            9: 75,
            10: 85,
            11: 95,
        }
        self.n_contexts: int = n_contexts

        self.biased_context: bool = biased_context

        if allowed_contexts is None:
            self.allowed_contexts: List[int] = list(self.reward_locations_by_context.keys())
        else:
            print(
                f'Only {allowed_contexts} of total {self.n_contexts} are available to the agent. User is probably performing curriculum learning experments.'
            )
            self.allowed_contexts: List[int] = allowed_contexts

        self.context_override: Optional[int] = context_override
        if self.context_override is not None:
            assert allowed_contexts is None

        self.obs_context_override: Optional[int] = obs_context_override
        if self.obs_context_override is not None:
            assert self.context_override is not None, 'This is only useful when context override is set'

        self.max_indicator_steps = indicator_steps

        self.OBS_DTYPE = np.float32
        self.context_n_onehot: int = len(
            self.reward_locations_by_context
        )  # Number needed to encode the reward locations
        self.observation_space = spaces.Box(low=0, high=1000, shape=(4 + self.context_n_onehot,), dtype=self.OBS_DTYPE)

        self.trial_schedule_b_nb_p = trial_schedule_b_nb_p

        self.n_trials = len(trial_schedule_b_nb_p)
        try:
            self.divide_reward_by = self.trial_schedule_b_nb_p.count('nb') + self.trial_schedule_b_nb_p.count('b')
        except ZeroDivisionError:
            self.divide_reward_by = 0

        self.middle_length = middle_length
        self.blackbox_length = blackbox_length
        self.track_length = self.middle_length + (2 * self.blackbox_length)

        self.neverstop_min_speed: float = 3.0

        # State variables
        self.location = None
        self.indicator_timer = None
        self.reward_acquired = False
        self.trial = None
        self.trial_types = None
        self.trial_speed: None = None
        self.rz_start = None
        self.rz_end = None
        self.context = None
        self.stats_episode_reward = None

        self.reset()

    def get_random_velocity(self) -> float:
        """selects a randomised velocity"""
        if self.trial_speed == 'normal':
            return (
                np.clip((3 - 7) * np.random.normal() + 7, 0, 100) / 3
            )  # gives about 100 steps per trial (not episode!)
        elif self.trial_speed == 'neverstop':
            return np.clip(
                np.clip((3 - 7) * np.random.normal() + 7, 0, 100) / 3, a_min=self.neverstop_min_speed, a_max=100
            )
        if self.trial_speed == 'normal_lownoise':
            l = 6.75
            h = 7.25
            return np.clip((l - h) * np.random.normal() + h, 0, 100) / 3
        elif self.trial_speed == 'constant_1p0':
            return 1.0
        elif self.trial_speed == 'constant_1p5':
            return 1.5
        elif self.trial_speed == 'constant_2p5':
            return 2.5
        elif self.trial_speed == 'constant_3p5':
            return 3.5
        elif self.trial_speed == 'constant_4p0':
            return 4.0
        elif self.trial_speed == 'constant_4p5':
            return 4.5
        elif self.trial_speed == 'constant_5p0':
            return 5.0
        elif 'custom' in self.trial_speed:  # e.g. custom_2
            v: float = float(self.trial_speed.split('_')[1])
            print(f'Parsed speed {v} from {self.trial_speed}')
            return v
        elif self.trial_speed == 'fast':
            return (np.clip((3 - 7) * np.random.normal() + 7, 0, 100) / 3) * 2
        elif self.trial_speed == 'slow':
            return (np.clip((3 - 7) * np.random.normal() + 7, 0, 100) / 3) * 0.5
        elif self.trial_speed == 'halfzero':
            return np.random.choice([np.clip((3 - 7) * np.random.normal() + 7, 0, 100) / 3, 0])
        elif self.trial_speed == 'veryslow':
            return np.random.uniform(low=0.05, high=0.15) * 4  #
        else:
            raise ValueError('Invalid speed')

    def get_random_context(self) -> int:
        if self.context_override is None:
            available_contexts: List[int] = list(self.reward_locations_by_context.keys())[0 : self.n_contexts]

            available_contexts: List[int] = [c for c in available_contexts if c in self.allowed_contexts]

            if self.biased_context:
                if len(available_contexts) == 1:
                    return available_contexts[0]
                else:
                    return np.random.choice([available_contexts[-1], np.random.choice(available_contexts[:-1])])
            else:
                return np.random.choice(available_contexts)

        else:
            return self.context_override

    def soft_reset(self):
        self.location = 0
        self.indicator_timer = 0
        self.reward_acquired = False

        self.context: int = self.get_random_context()
        rz_location: int = self.reward_locations_by_context[self.context]

        self.rz_start: float = rz_location - (self.rz_length / 2)
        self.rz_end: float = rz_location + (self.rz_length / 2)

        if self.speed_mode == 'mixed':
            self.trial_speed: str = np.random.choice(['normal', 'fast', 'slow'])
        else:
            self.trial_speed: str = self.speed_mode

    def get_trial_type(self):
        return self.trial_types[self.trial]

    def is_probe_trial(self):
        return self.get_trial_type() == 'p'

    def is_beaconed_trial(self):
        return self.get_trial_type() == 'b'

    def reset(self, seed=None, options: Union[Dict[str, Any], None] = None):
        self.stats_episode_reward = defaultdict(float)
        self.trial = 0
        self.trial_types = np.random.choice(self.trial_schedule_b_nb_p, len(self.trial_schedule_b_nb_p), replace=False)
        self.soft_reset()

        if self.trial_speed == 'neverstop':
            velocity = self.neverstop_min_speed
        else:
            velocity = 0

        state, _, _, _, info = self.get_state(
            velocity=velocity,
            reward=0,
            done=False,
            indicator_timer=self.indicator_timer,
            in_blackbox=True,
            show_beacon=False,
            stopped=False,
            context=self.context,
        )
        return state, info

    def get_state(
        self,
        velocity: float,
        reward: float,
        done: bool,
        indicator_timer: int,
        in_blackbox: bool,
        show_beacon: bool,
        stopped: bool,
        context: int,
    ):
        truncated = False
        info_step = {
            'location': self.location,
            'indicator_timer': indicator_timer,
            'self_indicator_timer': self.indicator_timer,
            'reward_acquired': self.reward_acquired,
            'trial': self.trial,
            'trial_type': self.get_trial_type(),
            'beacon': show_beacon,
            'velocity': velocity,
            'stopped': stopped,
            'rz_start': self.rz_start,
            'rz_end': self.rz_end,
            'context': self.context,
        }
        assert context == self.context

        info_settings = {
            'trial_types': self.trial_types,
            'track_length': self.track_length,
            'middle_length': self.middle_length,
            'blackbox_length': self.blackbox_length,
        }

        info = {'step': info_step, 'settings': info_settings}

        if done:
            info['episode_extra_stats'] = {**self.stats_episode_reward}

        return self.obs(velocity, indicator_timer, in_blackbox, show_beacon, context), reward, done, truncated, info

    def obs(
        self, velocity: float, indicator_timer: int, in_blackbox: bool, show_beacon: bool, context: int
    ) -> np.array:
        if self.obs_context_override is not None:
            context = self.obs_context_override

        context_onehot: List[bool] = [False] * self.context_n_onehot
        context_onehot[context] = True

        return np.array(
            [velocity, 1 if indicator_timer > 0 else 0, int(in_blackbox), int(show_beacon)] + context_onehot,
            dtype=self.OBS_DTYPE,
        )

    def is_in_blackbox(self):
        first = self.blackbox_length >= self.location >= 0
        second = self.location >= (self.middle_length + self.blackbox_length)

        return first or second

    def step(self, action) -> t.Tuple[np.array, bool, int, dict]:
        stopped = action == 0
        if not stopped:
            velocity = self.get_random_velocity()
        else:
            if self.trial_speed == 'neverstop':
                velocity = self.neverstop_min_speed
            else:
                velocity = 0
        self.location += velocity

        reached_end = self.location >= self.track_length
        if reached_end:
            if self.trial == (self.n_trials - 1):
                done = True
            else:
                done = False
                self.trial += 1
        else:
            done = False

        in_rz = self.rz_start <= self.location <= self.rz_end

        if stopped:
            if in_rz and not self.reward_acquired:
                if self.is_probe_trial():
                    reward = -1
                else:
                    reward = 100
                    self.indicator_timer = self.max_indicator_steps
                self.reward_acquired = True
            else:
                reward = -1
        else:
            reward = 0

        show_beacon = self.is_beaconed_trial() and in_rz

        try:
            reward = reward / self.divide_reward_by
        except ZeroDivisionError:
            reward = 0

        if reached_end and not done:
            self.soft_reset()

        state = self.get_state(
            velocity,
            reward=reward,
            done=done,
            indicator_timer=self.indicator_timer,
            in_blackbox=self.is_in_blackbox(),
            show_beacon=show_beacon,
            stopped=stopped,
            context=self.context,
        )

        self.indicator_timer -= 1

        self.stats_episode_reward['all'] += reward
        self.stats_episode_reward[self.get_trial_type()] += reward

        return state

    def render(self, dummy='dummy'):
        plt.clf()

        ax = plt.gca()

        ax.axvspan(self.rz_start, self.rz_end, alpha=0.15, color='red' if self.indicator_timer > -1 else 'green')

        ax.axvspan(0, 30, alpha=0.8 if self.is_in_blackbox() else 0.15, color='black')
        ax.axvspan(60 + 60 + 20 + 30, 200, alpha=0.8 if self.is_in_blackbox() else 0.15, color='black')
        plt.scatter(x=self.location, y=1)

        plt.xlim(0, 200)

        plt.draw()
        plt.pause(0.000001)
