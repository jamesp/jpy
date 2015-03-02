#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Memoize decorator for caching expensive function calls."""

import functools

def memoize(obj):
    """A function decorator for caching expensive to compute function calls.

    If a function has no side effects but is computationally expensive to
    compute, it can be memoized.

        @memoize
        def cpu_intensive_fn(a,b,c):
            ...
            return result

    When the function is called for the first time with arguments a, b, c it
    is run as normal, but the return value is stored as well as returned.
    Then on future calls with the same argument values, the stored value is
    returned immediately without recalculation.  Requires the arguments to be
    hashable.
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer
