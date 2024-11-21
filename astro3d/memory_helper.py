from itertools import repeat
from string import ascii_letters
from functools import partial
from multiprocessing import shared_memory
import multiprocessing
import numpy as np
import time


def getname():
    """
    Generates a random string of 15 ascii letters.
    The function selects 15 random characters from the `ascii_letters` set and 
    concatenates them into a single string.
    Returns:
        str: A random string of 15 characters.
    """

    return ''.join([ascii_letters[i] for i in np.random.choice(len(ascii_letters), 15)])


def set_up_shared_memory(arr=None, name=None):
    """
    Set up a shared memory array.

    This function creates a shared memory block and returns its name. The shared memory
    can be used to share data between different processes.

    Parameters:
    arr (numpy.ndarray, optional): Existing NumPy array to be shared.

    Returns:
    str: The name of the shared memory block.

    """

    name = name or getname()

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
    arr = b  # keeps this one to avoid garbage collection

    # Close the shared memory, but don't unlink it
    shm.close()

    return name


def get_shared_array(name, shape, dtype=np.float64):
    """
    Retrieve a shared memory array.

    This function opens an existing shared memory block by name and returns
    a NumPy array that uses this shared memory as its buffer.

    Parameters:
    name (str): The name of the shared memory block.
    shape (tuple): The shape of the array to be created from the shared memory.

    Returns:
    tuple: A tuple containing the shared memory object and the NumPy array.
    """

    # Open the existing shared memory
    shm = shared_memory.SharedMemory(name=name)

    # Load the array from shared memory
    shared_array = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=shm.buf)

    # to avoid garbage collection, we need to return both
    return shm, shared_array


def _work(i, name, shape):
    """
    An example on how to perform operations on a shared memory array.

    This function opens an existing shared memory array, retrieves its ID and the value at the 
    first position, pretends to work for 2 seconds, and then closes the shared memory.

    Args:
        i (int): An index or identifier (like a task ID, but not used here).
        name (str): The name of the shared memory block.
        shape (tuple): The shape of the array stored in the shared memory.

    Returns:
        tuple: A tuple containing the ID of the array's data buffer and the value at the first 
               position of the array.
    """

    # Open the existing shared memory
    shm, arr = get_shared_array(name, shape)

    # save the ID and value
    time.sleep(2)
    _id = id(shm.buf)
    val = arr[0, 0, 0]
    arr[0, 0, 0] += 1

    # Close the shared memory
    shm.close()

    return _id, val


def test_shared_memory(n_proc=8):
    """
    Test function for shared memory operations.
    This function sets up a shared memory array, distributes work across
    multiple processes using a multiprocessing pool, and then retrieves
    and prints the results. Finally, it unlinks the shared memory.
    Steps:
    1. Set up a shared memory array with the specified shape.
    2. Create a multiprocessing pool with 4 processes.
    3. Map the `_work` function to the pool, passing the shared memory name and shape.
    4. Print the first and second elements of the results from each process.
    5. Retrieve the shared memory array one last time to close it.
    6. Unlink the shared memory to clean up resources.
    """

    N = 500
    shape = [N, N, N]
    myarr = np.random.rand(*shape)
    myarr[0, 0, 0] = 0.0
    name = set_up_shared_memory(arr=myarr)
    print(f'shared array name = {name}')
    print(f'shared array size: {myarr.nbytes / (1024)**3:.3f} GB')

    with multiprocessing.Pool(n_proc) as p:
        res = p.starmap(_work, zip(range(n_proc), repeat(name), repeat(shape)))

    for r in res:
        print(f'{r[0]}\t{r[1]}')

    if not np.all(np.sort([r[0] for r in res]) == np.arange(n_proc)):
        print('not ', end='')
    print('all good!')

    # get the array one last time to close it
    shm, _ = get_shared_array(name, shape)
    shm.close()
    shm.unlink()


if __name__ == '__main__':
    test_shared_memory()
