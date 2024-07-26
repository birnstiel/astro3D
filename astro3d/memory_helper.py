import time
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

from functools import partial

from string import printable


def getname():
    return ''.join([printable[i] for i in np.random.choice(len(printable), 15)])


def set_up_shared_memory(shape, name=None, dtype=None):
    """
    Sets up shared memory of the given shape and name, will asign a random name otherwise.
    """

    name = name or getname()

    # create a NumPy array
    arr = np.zeros(shape, dtype=None)
    print(f'shared array size: {arr.nbytes / (1024)**3:.3f} GB')

    # create the shared memory and give it a name
    shm = shared_memory.SharedMemory(
        create=True,
        size=arr.nbytes,
        name=name)

    # create a NumPy array backed by shared memory
    b = np.ndarray(
        arr.shape,
        dtype=arr.dtype,
        buffer=shm.buf)

    # Copy the original data into shared memory
    b[:] = arr[:]
    arr = b

    return name


def get_shared_array(name, shape):

    # Open the existing shared memory
    shm = shared_memory.SharedMemory(name=name)

    # Load the array from shared memory
    shared_array = np.ndarray(
        shape=shape,
        dtype=np.float64,
        buffer=shm.buf)

    # to avoid garbage collection, we need to return both
    return shm, shared_array


def _work(i, name, shape):

    # Open the existing shared memory
    shm, arr = get_shared_array(name, shape)

    # save the ID and value
    time.sleep(2)
    _id = id(arr.data)
    val = arr[0, 0, 0]

    # Close the shared memory
    shm.close()

    return _id, val


def test_shared_memory():
    N = 100
    shape = [N, N, N]
    name = set_up_shared_memory(shape)

    p = multiprocessing.Pool(4)
    res = p.map(partial(_work, name=name, shape=shape), range(4))

    print([r[0] for r in res])
    print([r[1] for r in res])

    # get the array one last time to close it
    shm, arr = get_shared_array(name, shape)
    shm.unlink()

    p.close()


if __name__ == '__main__':
    test_shared_memory()
