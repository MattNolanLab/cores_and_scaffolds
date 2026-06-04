from typing import List, Tuple

from gym.envs.registration import register

for ctx in range(2):
    register(
        id=f'LinearNavigationContextualStatic{ctx}-v2',
        entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
        kwargs={'context_override': ctx},
    )


def make_allowed_contexts_str(contexts: List[int]) -> str:
    return 'c'.join([str(c) for c in contexts])


def make_curriculum_contexts() -> List[str]:
    """TODO: Returns only the BiasedAllowed"""
    envs: List[str] = []

    max_contexts: int = 12
    for allowed_context in range(0, max_contexts):
        allowed_contexts: List[int] = list(range(0, allowed_context + 1))
        allowed_context_str: str = make_allowed_contexts_str(allowed_contexts)

        env_id: str = f'LinearNavigationContextual{max_contexts}Allowed{allowed_context_str}-v2'
        register(
            id=env_id,
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': max_contexts, 'allowed_contexts': allowed_contexts},
        )

        env_id: str = f'LinearNavigationContextualFixed{max_contexts}Allowed{allowed_context_str}-v2'
        register(
            id=env_id,
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualFixedV2',
            kwargs={'n_contexts': max_contexts, 'allowed_contexts': allowed_contexts},
        )

        env_id: str = f'LinearNavigationNormalLowNoiseContextual{max_contexts}Allowed{allowed_context_str}-v2'
        register(
            id=env_id,
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': max_contexts, 'allowed_contexts': allowed_contexts, 'speed': 'normal_lownoise'},
        )

        register(
            id=f'LinearNavigationContextual{max_contexts}SingleAllowed{allowed_context_str}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={
                'n_contexts': max_contexts,
                'allowed_contexts': allowed_contexts,
                'biased_context': True,
                'trial_schedule_b_nb_p': ('nb',),
            },
        )

        env_id = f'LinearNavigationContextual{max_contexts}BiasedAllowed{allowed_context_str}-v2'
        envs.append(env_id)
        register(
            id=env_id,
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': max_contexts, 'allowed_contexts': allowed_contexts, 'biased_context': True},
        )

        register(
            id=f'LinearNavigationContextual{max_contexts}Constant2p5BiasedAllowed{allowed_context_str}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={
                'n_contexts': max_contexts,
                'allowed_contexts': allowed_contexts,
                'biased_context': True,
                'speed': 'constant_2p5',
            },
        )
    return envs


ENVS_CURRICULUM_CONTEXTUAL_BIASED = make_curriculum_contexts()


for n_ctx in range(2, 12 + 1):
    register(
        id=f'LinearNavigationContextual{n_ctx}-v2',
        entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
        kwargs={'n_contexts': n_ctx, 'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb')},
    )

    register(
        id=f'LinearNavigationNormalLowNoiseContextual{n_ctx}-v2',
        entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
        kwargs={
            'n_contexts': n_ctx,
            'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
            'speed': 'normal_lownoise',
        },
    )
    for ctx in range(n_ctx):
        register(
            id=f'LinearNavigationContextual{n_ctx}Static{ctx}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={
                'n_contexts': n_ctx,
                'context_override': ctx,
                'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
            },
        )

        register(
            id=f'LinearNavigationProbeContextual{n_ctx}Static{ctx}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': n_ctx, 'context_override': ctx, 'trial_schedule_b_nb_p': ('p',)},
        )

        register(
            id=f'LinearNavigationSingleNbContextual{n_ctx}Static{ctx}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': n_ctx, 'context_override': ctx, 'trial_schedule_b_nb_p': ('nb',)},
        )

        env_id: str = f'LinearNavigationContextualFixed{n_ctx}Static{ctx}-v2'
        register(
            id=env_id,
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualFixedV2',
            kwargs={
                'n_contexts': n_ctx,
                'context_override': ctx,
                'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
            },
        )

        for speed in [
            'normal',
            'constant_1p0',
            'constant_1p5',
            'constant_2p5',
            'constant_3p5',
            'constant_4p0',
            'constant_4p5',
            'constant_5p0',
            'fast',
            'slow',
            'veryslow',
            'custom_1',
            'custom_2',
            'custom_3',
            'custom_4',
            'halfzero',
            'neverstop',
        ]:
            register(
                id=f'LinearNavigationContextual{n_ctx}Static{ctx}Speed{speed}-v2',
                entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
                kwargs={
                    'n_contexts': n_ctx,
                    'context_override': ctx,
                    'speed': speed,
                    'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
                },
            )

            register(
                id=f'LinearNavigationProbeContextual{n_ctx}Static{ctx}Speed{speed}-v2',
                entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
                kwargs={
                    'n_contexts': n_ctx,
                    'context_override': ctx,
                    'speed': speed,
                    'trial_schedule_b_nb_p': ('p', 'p', 'p', 'p', 'p', 'p'),
                },
            )

            register(
                id=f'LinearNavigationSingleNbContextual{n_ctx}Static{ctx}Speed{speed}-v2',
                entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
                kwargs={'n_contexts': n_ctx, 'context_override': ctx, 'trial_schedule_b_nb_p': ('nb',), 'speed': speed},
            )

        for obs_ctx in range(n_ctx):
            register(
                id=f'LinearNavigationContextual{n_ctx}Static{ctx}Obs{obs_ctx}-v2',
                entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
                kwargs={
                    'n_contexts': n_ctx,
                    'context_override': ctx,
                    'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
                    'obs_context_override': obs_ctx,
                },
            )

        register(
            id=f'LinearNavigationNormalLowNoiseContextual{n_ctx}Static{ctx}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={
                'n_contexts': n_ctx,
                'trial_schedule_b_nb_p': ('b', 'nb', 'b', 'nb', 'b', 'nb'),
                'speed': 'normal_lownoise',
                'context_override': ctx,
            },
        )

        register(  # TODO obselete this
            id=f'LinearNavigationContextual{n_ctx}Constant2p5Static{ctx}-v2',
            entry_point='gym_linearnavigation.envs:LinearNavigationContextualV2',
            kwargs={'n_contexts': n_ctx, 'context_override': ctx, 'speed': 'constant_2p5'},
        )
