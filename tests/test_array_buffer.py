import json
import pickle

import numpy as np
from pyprof import profile, Profiler

from ndarraybuffer import ArrayBuffer


def test_array_buffer_push_pop() -> None:
    arr = ArrayBuffer()
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
    arr = ArrayBuffer(max_len=10)
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
    arr = ArrayBuffer(max_len=10)
    arr.extend(np.arange(10))
    assert [_ for _ in arr] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_dtype() -> None:
    arr = ArrayBuffer(dtype=np.int32)
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
    arr = ArrayBuffer(max_len=10, dtype=np.float64)
    arr.extend(np.arange(10))
    rec: ArrayBuffer = pickle.loads(pickle.dumps(arr))
    assert rec.max_len == 10
    assert rec.dtype == np.float64
    assert np.array_equal(rec, np.arange(10))


def test_json_serialization() -> None:
    arr = ArrayBuffer(max_len=10, dtype=np.float64)
    arr.extend(np.arange(10))
    state_dict = arr.state_dict()
    rec = ArrayBuffer.load(json.loads(json.dumps(state_dict)))
    assert rec.max_len == 10
    assert rec.dtype == np.float64
    assert np.array_equal(rec, np.arange(10))


def test_efficiency() -> None:
    arr = ArrayBuffer(max_len=10, dtype=np.float64)

    with profile(""):
        for i in range(10000):
            profile(arr.append)(i)
            assert len(arr) <= 10
    assert Profiler.get("//ArrayBuffer.append").average < 1e-4


def test_conversion_to_ndarray() -> None:
    arr = ArrayBuffer()
    arr.extend(np.arange(10))
    assert np.array_equal(np.asarray(arr), np.arange(10))
    assert np.array_equal(np.asarray(arr, dtype=np.int_), np.arange(10))
    assert np.array_equal(np.asarray(arr, dtype=np.float_), np.arange(10))


def test_array_does_not_copy() -> None:
    arr = ArrayBuffer()
    arr.extend(np.arange(10))
    a = np.asarray(arr)
    a[0] = -1
    assert np.array_equal(arr, a)
    assert arr[0] == -1


def test_operations() -> None:
    arr = ArrayBuffer()
    arr.extend(np.arange(10))
    assert np.array_equal(arr + 1, np.arange(1, 11))
    assert np.array_equal(arr - 1, np.arange(-1, 9))
    assert np.array_equal(arr * 2, np.arange(10) * 2)
    assert np.array_equal(arr / 2, np.arange(10) / 2)
    assert np.array_equal(arr // 2, np.arange(10) // 2)
    arr += 1
    assert isinstance(arr, ArrayBuffer)
    assert np.array_equal(arr, np.arange(1, 11))
    arr *= 2
    assert isinstance(arr, ArrayBuffer)
    assert np.array_equal(arr, np.arange(1, 11) * 2)
    arr -= np.arange(2, 12)
    assert isinstance(arr, ArrayBuffer)
    assert np.array_equal(arr, np.arange(10))

    assert len(arr[np.where(arr < 2)]) == 2
    assert len(arr[np.where(arr > 2)]) == 7
    assert len(arr[np.where(arr == 2)]) == 1
    assert len(arr[np.where(arr <= 2)]) == 3
    assert len(arr[np.where(arr >= 2)]) == 8
    assert len(arr[arr % 2 == 0]) == 5
    assert arr @ np.ones(10) == 45

    assert np.array_equal(-arr, -np.arange(10))


def test_to_list() -> None:
    arr = ArrayBuffer()
    arr.extend(np.arange(10))
    assert list(arr) == list(range(10))


def test_setitem() -> None:
    arr = ArrayBuffer(dtype=np.int64)
    arr.extend(np.arange(10))
    arr[0] += 1
    arr[-1] = arr[-1] + 1
    assert arr[0] == 1
    assert arr[-1] == 10


def test_extend_large():
    arr = ArrayBuffer(dtype=np.int64, max_len=10)
    arr.extend(np.arange(100))
    assert np.array_equal(arr, np.arange(90, 100))
    arr.extendleft(np.arange(100))
    assert np.array_equal(arr, np.arange(0, 10))
