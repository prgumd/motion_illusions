###############################################################################
#
# File: rate_limit.py
#
# Accurate rate limiter
#
# History:
# 02-13-20 - Levi Burner - Created file
#
###############################################################################

import time
import numpy as np

class RateLimit(object):
    def __init__(self, limit_hz=None, limit_period_s=None, collect_stats=False, num_stat_elements=100):
        self._t_last = time.perf_counter()

        if limit_hz is not None:
            self._min_period_s = (1.0 / limit_hz)

        if limit_period_s is not None:
            self._min_period_s = limit_period_s

        self._collect_stats = collect_stats
        self._num_stat_elements = num_stat_elements

        if self._collect_stats:
            self._stat_index = 0
            self._past_timestamps = np.zeros((num_stat_elements, 3))

    def sleep(self):
        t = time.perf_counter()
        client_delta_s = t - self._t_last
        if client_delta_s < self._min_period_s:
            sleep_duration = self._min_period_s - client_delta_s
            time.sleep(sleep_duration)

            true_delta_s = time.perf_counter() - self._t_last
            self._t_last = self._t_last + self._min_period_s
        else:
            sleep_duration = 0.0
            t = time.perf_counter()
            true_delta_s = t - self._t_last
            self._t_last = t

        if self._collect_stats:
            self._past_timestamps[self._stat_index, :] = (client_delta_s, sleep_duration, true_delta_s)
            self._stat_index = (self._stat_index + 1) % self._num_stat_elements

    def print_stats(self):
        avg_ms = 1000*np.average(self._past_timestamps, axis=0)
        std_dev_ms = 1000*np.sqrt(np.var(self._past_timestamps, axis=0))
        print('Client avg: %d ms std dev: %d ms' % (avg_ms[0], std_dev_ms[0]))
        print('Sleep avg: %d ms std dev: %d ms' % (avg_ms[1], std_dev_ms[1]))
        print('True avg: %d ms std dev: %d ms' % (avg_ms[2], std_dev_ms[2]))
