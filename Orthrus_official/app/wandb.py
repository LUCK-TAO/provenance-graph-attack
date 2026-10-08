"""Offline no-op shim for Orthrus runs that do not enable Weights & Biases."""


class _Config:
    def update(self, *_args, **_kwargs):
        return None


class Image:
    def __init__(self, value, *_args, **_kwargs):
        self.value = value


config = _Config()


def init(*_args, **_kwargs):
    return None


def log(*_args, **_kwargs):
    return None


def finish(*_args, **_kwargs):
    return None
