import json
import pickle

import numpy as np
from numpy.typing import NDArray
from pyprof import profile, Profiler

from ndarraybuffer import empty, ArrayBuffer, load


def test_array_buffer_push_pop() -> None:
    arr: ArrayBuffer = empty(np.float64)
    assert len(arr) == 0
    assert arr.max_len is None
    assert arr.shape == (0,)
    arr.extend(np.arange(100))
    assert arr[0] == 0
    assert len(arr) == 100
    assert arr.max_len is None
    assert arr.shape == (100,)
    assert np.array_equal(arr, np.arange(100))

    arr.append(100)
    assert len(arr) == 101
    assert arr.shape == (101,)
    assert np.array_equal(arr, np.arange(101))

    arr.pop(1)
    assert len(arr) == 100
    assert arr.shape == (100,)
    assert np.array_equal(arr, np.arange(100))

    arr.popleft(1)
    assert len(arr) == 99
    assert arr.shape == (99,)
    assert np.array_equal(arr, np.arange(1, 100))

    arr.appendleft(0)
    assert len(arr) == 100
    assert arr.shape == (100,)
    assert np.array_equal(arr, np.arange(100))

    for i in range(100, 10000):
        arr.extend([i])
    assert len(arr) == 10000
    assert np.array_equal(arr, np.arange(10000))

    arr[-1] = 0
    assert np.array_equal(arr[:-1], np.arange(10000 - 1))
    assert arr[-1] == 0

    arr.clear()
    assert len(arr) == 0


def test_max_len() -> None:
    arr = ArrayBuffer[np.int32](max_len=10)
    arr.extend(np.arange(10))
    assert len(arr) == 10
    assert np.array_equal(arr, np.arange(10))
    arr.append(10)
    assert len(arr) == 10
    assert np.array_equal(arr, np.arange(1, 11))
    arr.appendleft(0)
    assert len(arr) == 10
    assert np.array_equal(arr, np.arange(0, 10))


def test_iter() -> None:
    arr = ArrayBuffer[np.int32](max_len=10)
    arr.extend(np.arange(10))
    assert [_ for _ in arr] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_dtype() -> None:
    arr = ArrayBuffer[np.int32](dtype=np.int32)
    assert arr.nbytes == 2048 * 4
    assert arr.dtype == np.int32
    assert arr.itemsize == 4
    arr.append(1.1)
    assert np.array_equal(arr, [1])

    arr = ArrayBuffer(dtype=np.float64, init_shape=1024)
    assert arr.nbytes == 1024 * 8
    assert arr.dtype == np.float64
    assert arr.itemsize == 8
    arr.append(1.1)
    assert not np.array_equal(arr, [1])
    assert np.array_equal(arr, [1.1])


def test_pickle() -> None:
    arr = ArrayBuffer[np.float64](max_len=10, dtype=np.float64)
    arr.extend(np.arange(10))
    rec: ArrayBuffer = pickle.loads(pickle.dumps(arr))
    assert rec.max_len == 10
    assert rec.dtype == np.float64
    assert np.array_equal(rec, np.arange(10))


def test_json_serialization() -> None:
    arr = ArrayBuffer[np.float64](max_len=10, dtype=np.float64)
    arr.extend(np.arange(10))
    state_dict = arr.state_dict()
    rec = ArrayBuffer.load(json.loads(json.dumps(state_dict)))
    assert rec.max_len == 10
    assert rec.dtype == np.float64
    assert np.array_equal(rec, np.arange(10))


def test_json_serialization_astype() -> None:
    arr = ArrayBuffer[np.float64](max_len=10, dtype=np.float64)
    arr.extend(np.arange(10))
    assert arr[0].dtype == np.dtype(np.float64)
    state_dict = arr.state_dict()
    rec = load(json.loads(json.dumps(state_dict)))
    assert rec.max_len == 10
    assert rec.dtype == np.float64
    assert rec[0].dtype ==  np.dtype(np.float64)
    assert np.array_equal(rec, np.arange(10))

    rec2 = load(json.loads(json.dumps(state_dict)), dtype=np.int64)
    assert rec2.max_len == 10
    assert rec2.dtype == np.int64
    assert rec2[0].dtype ==  np.dtype(np.int64)
    assert np.array_equal(rec2, np.arange(10))


def test_efficiency() -> None:
    arr = ArrayBuffer[np.float64](max_len=10, dtype=np.float64)

    with profile(""):
        for i in range(10000):
            profile(arr.append)(i)
            assert len(arr) <= 10
    assert Profiler.get("//_ArrayBuffer.append").average < 1e-4


def test_conversion_to_ndarray() -> None:
    arr = ArrayBuffer[np.float64]()
    arr.extend(np.arange(10))
    assert np.array_equal(np.asarray(arr), np.arange(10))
    assert np.array_equal(np.asarray(arr, dtype=np.int64), np.arange(10))
    assert np.array_equal(np.asarray(arr, dtype=np.float64), np.arange(10))


def test_array_does_not_copy() -> None:
    arr = ArrayBuffer[np.float64]()
    arr.extend(np.arange(10))
    a: NDArray = np.asarray(arr)
    a[0] = -1
    assert np.array_equal(arr, a)
    assert arr[0] == -1


def test_operations() -> None:
    arr = ArrayBuffer[np.float64]()
    arr.extend(np.arange(10))
    assert np.array_equal(arr + 1, np.arange(1, 11))
    assert np.array_equal(arr - 1, np.arange(-1, 9))
    assert np.array_equal(arr * 2, np.arange(10) * 2)
    assert np.array_equal(arr / 2, np.arange(10) / 2)
    assert np.array_equal(arr // 2, np.arange(10) // 2)
    arr += 1
    assert isinstance(arr, type(empty(arr.dtype)))
    assert np.array_equal(arr, np.arange(1, 11))
    arr *= 2
    assert isinstance(arr, type(empty(arr.dtype)))
    assert np.array_equal(arr, np.arange(1, 11) * 2)
    arr -= np.arange(2, 12)
    assert isinstance(arr, type(empty(arr.dtype)))
    assert np.array_equal(arr, np.arange(10))

    assert len(arr[np.where(arr < 2)]) == 2
    assert len(arr[np.where(arr > 2)]) == 7
    assert len(arr[np.where(arr == 2)]) == 1
    assert len(arr[np.where(arr <= 2)]) == 3
    assert len(arr[np.where(arr >= 2)]) == 8
    x = np.array([2, 3, 4])
    assert len(arr[arr % 2 == 0]) == 5
    assert arr @ np.ones(10) == 45

    assert np.array_equal(-arr, -np.arange(10))


def test_to_list() -> None:
    arr = ArrayBuffer[np.float64]()
    arr.extend(np.arange(10))
    assert list(arr) == list(range(10))


def test_setitem() -> None:
    arr = ArrayBuffer[np.float64](dtype=np.int64)
    arr.extend(np.arange(10))
    arr[0] += 1
    arr[-1] = arr[-1] + 1
    assert arr[0] == 1
    assert arr[-1] == 10


def test_extend_large() -> None:
    arr = ArrayBuffer[np.float64](dtype=np.int64, max_len=10)
    arr.extend(np.arange(100))
    assert np.array_equal(arr, np.arange(90, 100))
    arr.extendleft(np.arange(100))
    assert np.array_equal(arr, np.arange(0, 10))


def test_empty_array_op() -> None:
    arr = ArrayBuffer(dtype=np.int64)
    assert len(arr == 0) == 0
    assert len(arr != 0) == 0
    assert len(arr + 0) == 0


def test_multidim_array() -> None:
    # Test initialization and basic operations with multidimensional arrays
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 2, 3))
    assert arr.shape == (0, 2, 3)

    data = np.ones((5, 2, 3))
    arr.extend(data)
    assert arr.shape == (5, 2, 3)

    # Test indexing
    assert arr[0].shape == (2, 3)
    assert arr[0, 0].shape == (3,)
    assert arr[0, 0, 0].dtype == np.dtype(np.float64)

    # Test slicing
    assert arr[:2].shape == (2, 2, 3)
    assert arr[:, 0].shape == (5, 3)
    assert arr[..., 0].shape == (5, 2)

    # Test advanced indexing
    idx = np.array([True, False, True, False, True])
    assert arr[idx].shape == (3, 2, 3)

    # Test operations
    arr *= 2
    assert np.array_equal(arr, data * 2)

    # Test append/extend
    new_data = np.zeros((1, 2, 3))
    arr.extend(new_data)
    assert arr.shape == (6, 2, 3)


def test_complex_indexing() -> None:
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 2))
    data = np.arange(10 * 2).reshape(-1, 2)
    arr.extend(data)

    # Test boolean indexing
    mask = arr[:, 0] > 10
    assert arr[mask].shape == (4, 2)

    # Test integer array indexing
    idx = np.array([1, 3, 5])
    assert arr[idx].shape == (3, 2)

    # Test combined indexing
    assert arr[2:5, 1].shape == (3,)

    # Test setting values
    arr[2:4] = np.ones((2, 2))
    assert np.array_equal(arr[2:4], np.ones((2, 2)))


def test_type_operations() -> None:
    # Test operations with different types
    arr = ArrayBuffer(dtype=np.float32)
    arr.extend(np.arange(10, dtype=np.float32))

    # Integer operation should preserve float32
    result = arr * 2
    assert result.dtype == np.float32

    # Float operation
    result = arr / 2.0
    assert result.dtype == np.float32

    # Test with bool arrays
    bool_arr = arr > 5
    assert bool_arr.dtype == np.bool_

    # Test comparison operations
    assert (arr > 5).dtype == np.bool_
    assert (arr >= 5).dtype == np.bool_
    assert (arr < 5).dtype == np.bool_
    assert (arr <= 5).dtype == np.bool_
    assert (arr == 5).dtype == np.bool_
    assert (arr != 5).dtype == np.bool_


def test_reshape_operations() -> None:
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 2))
    data = np.arange(10 * 2).reshape(-1, 2)
    arr.extend(data)

    # Test array interface with reshape
    nd_arr: NDArray = np.asarray(arr)
    reshaped = nd_arr.reshape(-1)
    assert reshaped.shape == (20,)

    # Test operations after reshape
    result = reshaped + 1
    assert result.shape == (20,)


def test_matmul_operations() -> None:
    # Test matrix multiplication with different shapes
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 3))
    data = np.ones((5, 3))
    arr.extend(data)

    # Matrix multiplication with vector
    vector = np.ones(3)
    result = arr @ vector
    assert result.shape == (5,)

    # Matrix multiplication with matrix
    matrix = np.ones((3, 2))
    result = arr @ matrix
    assert result.shape == (5, 2)


def test_empty_operations() -> None:
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 2, 3))

    # Operations on empty array should preserve shape
    assert arr.shape == (0, 2, 3)
    result = arr + 1
    assert result.shape == (0, 2, 3)

    # Boolean operations
    assert (arr > 0).shape == (0, 2, 3)

    # Matrix multiplication
    try:
        _ = arr @ np.ones((3, 2))
    except ValueError:
        pass  # Expected error for empty array matmul


def test_broadcasting() -> None:
    arr = ArrayBuffer(dtype=np.float64, init_shape=(1024, 2, 3))
    data = np.ones((5, 2, 3))
    arr.extend(data)

    # Broadcasting scalar
    result = arr + 1
    assert result.shape == (5, 2, 3)

    # Broadcasting array
    result = arr + np.ones((2, 3))
    assert result.shape == (5, 2, 3)

    # In-place broadcasting
    arr += np.ones((2, 3))
    assert arr.shape == (5, 2, 3)