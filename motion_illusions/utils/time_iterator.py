###############################################################################
#
# File: time_iterator.py
#
# An object that iterate over wall clock time or a fixed timestep
#
# History:
# 07-27-20 - Levi Burner - Created file
#
###############################################################################

class TimeIterator(object):
    # Leave sim_rate_hz as None to use wall time
    def __init__(self, sim_rate_hz=None):
        self._sim_rate_hz = sim_rate_hz
        self._t = 0.0
    def __iter__(self):
        return self
    def __next__(self):
        if self._sim_rate_hz is None:
            return time.time()
        else:
            self._t += 1.0 / self._sim_rate_hz
            return self._t
