# Import modules
import carla

import random
import time


# modules
def array_to_vector2D(array):
    vector = carla.Vector2D(array[0], array[1])
    return vector

    
def array_to_vector3D(array):
    vector = carla.Vector3D(array[0], array[1], array[2])
    return vector


def sleep_random_time(start=2.0, end=6.0, verbose=False):
    '''
    Sleeps the thread for some random time between the provided range.

    Parameters:
    start : float
        The minimum time in which to sleep.
    end : float
        The maximum time in which to sleep.
    verbose : bool
        Used to determine whether some information should be displayed.
    '''
    sleep_time = random.uniform(start, end)

    if verbose:
        print('Sleeping for', sleep_time, 'seconds.')

    time.sleep(sleep_time)
