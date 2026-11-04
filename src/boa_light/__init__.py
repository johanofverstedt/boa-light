"""BOA_light: fast, lightweight Body and Organ Analysis (BOA) / TotalSegmentator runner."""

__version__ = "0.1.0"

__all__ = ["__version__", "analyze", "Options"]


def __getattr__(name):  # lazy import so `import boa_light` stays cheap
    if name in ("analyze", "Options"):
        from boa_light import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
