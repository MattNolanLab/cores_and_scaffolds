FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

RUN apt-get update --fix-missing
RUN apt-get install -y python3 git fonts-liberation

RUN mkdir /code
WORKDIR /code
COPY requirements.txt .
COPY libraries ./libraries/
COPY setup.py .
RUN mkdir -p ./src/sf

RUN uv pip install --system -e libraries/sample-factory
RUN uv pip install --system -e libraries/gym_linearnavigation
RUN uv pip install --system -e .

RUN uv pip install --system -r requirements.txt

RUN DEBIAN_FRONTEND='noninteractive' apt-get install --fix-missing -y python3-opencv

# https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y ttf-mscorefonts-installer gcc python3-dev

ENV PYTHONWARNINGS=ignore::DeprecationWarning:gym.utils.passive_env_checker
