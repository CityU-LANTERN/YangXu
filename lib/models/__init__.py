try:
    import timm
    _has_timm = True
except ModuleNotFoundError:
    _has_timm = False