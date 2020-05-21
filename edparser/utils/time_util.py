# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-27 00:01
import datetime
import time


def human_time_delta(days, hours, minutes, seconds, delimiter=' ') -> str:
    units = locals().copy()
    units.pop('delimiter')
    non_zero = False
    result = []
    for key, val in sorted(units.items()):
        append = False
        if non_zero:
            append = True
        elif val:
            non_zero = True
            append = True
        if append:
            result.append('{} {}'.format(val, key[0]))
    if not non_zero:
        return '0 s'
    return delimiter.join(result)


def seconds_to_time_delta(seconds):
    seconds = round(seconds)
    days = seconds // 86400
    hours = seconds // 3600 % 24
    minutes = seconds // 60 % 60
    seconds = seconds % 60
    return days, hours, minutes, seconds


def report_time_delta(seconds, human=True):
    days, hours, minutes, seconds = seconds_to_time_delta(seconds)
    if human:
        return human_time_delta(days, hours, minutes, seconds)
    return days, hours, minutes, seconds


class HumanTimeDelta(object):

    def __init__(self, delta_seconds) -> None:
        super().__init__()
        self.delta_seconds = delta_seconds

    def report(self, human=True):
        return report_time_delta(self.delta_seconds, human)

    def __str__(self) -> str:
        return self.report(human=True)

    def __truediv__(self, scalar):
        return HumanTimeDelta(self.delta_seconds / scalar)


class CountdownTimer(object):

    def __init__(self, total: int) -> None:
        super().__init__()
        self.total = total
        self.current = 0
        self.start = time.time()
        self.finished_in = None

    def update(self, n=1):
        self.current += n
        self.current = min(self.total, self.current)

    def ratio(self) -> str:
        return f'{self.current}/{self.total}'

    def eta(self) -> float:
        if self.finished_in:
            eta = self.finished_in
        elif self.total == self.current:
            eta = time.time() - self.start
            self.finished_in = eta
        else:
            eta = (time.time() - self.start) / max(self.current, 0.1) * (self.total - self.current)

        return eta

    def eta_human(self) -> str:
        return human_time_delta(*seconds_to_time_delta(self.eta()))

    def stop(self):
        if not self.finished_in:
            self.finished_in = time.time() - self.start
            self.total = self.current


class Timer(object):
    def __init__(self) -> None:
        self.last = time.time()

    def start(self):
        self.last = time.time()

    def stop(self) -> HumanTimeDelta:
        now = time.time()
        seconds = now - self.last
        self.last = now
        return HumanTimeDelta(seconds)


def now_human(year='y'):
    now = datetime.datetime.now()
    return now.strftime(f"%{year}-%m-%d %H:%M:%S")


def now_datetime():
    return now_human('Y')


def now_filename(fmt="%y%m%d_%H%M%S"):
    """
    Generate filename using current datetime, in 20180102_030405 format
    Returns
    -------

    """
    now = datetime.datetime.now()
    return now.strftime(fmt)
