import sys

import gym_linearnavigation  # noqa
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from sf.utils import register_custom_envs


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_envs()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
