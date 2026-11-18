"""Read run-time settings without editing source files.

`get_param("recon_target", "PLANT")` looks up, in order:

1. the private ROS parameter ``~recon_target`` (set with ``<param>`` in a launch
   file or ``_recon_target:=FRUIT`` on the ``rosrun`` command line),
2. the environment variable ``RECON_TARGET``,
3. the given default.

The result is cast to the type of the default. For list defaults, a
comma-separated string such as ``"1,2,3"`` is accepted.
"""

import os


def _cast(value, default):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, list):
        if isinstance(value, str):
            items = [v.strip() for v in value.split(",") if v.strip()]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        if default:
            elem_type = type(default[0])
            items = [elem_type(v) for v in items]
        return items
    if default is None:
        return value
    return type(default)(value)


def get_param(name, default):
    try:
        import rospy

        if rospy.core.is_initialized() and rospy.has_param("~" + name):
            return _cast(rospy.get_param("~" + name), default)
    except Exception:
        pass
    env = os.environ.get(name.upper())
    if env is not None and env != "":
        return _cast(env, default)
    return default
