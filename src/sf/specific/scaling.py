from typing import Dict, Tuple

import gym

N_CONTEXTS: int = 8
RZ_LENGTH: int = gym.make('LinearNavigationContextual-v2').rz_length
context_rzs: Dict[int, int] = gym.make('LinearNavigationContextual-v2').reward_locations_by_context
assert RZ_LENGTH == 10
RZ_MINS: Tuple[float] = tuple([float(rz - (RZ_LENGTH / 2)) for rz in context_rzs.values()][:N_CONTEXTS])
RZ_MAXES: Tuple[float] = tuple([float(rz + (RZ_LENGTH / 2)) for rz in context_rzs.values()][:N_CONTEXTS])
MIDDLE_CONTEXT: int = 3
