import enum
from typing import Iterator, Any, SupportsIndex

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike
from typing_extensions import Self, overload, TypeVar, Generic

__all__ = [
    "ArrayBuffer",
]


class Side(enum.IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


ScalarType = np.number | np.bool_ | bool | int | float
IndexArrayType = NDArray[np.bool_] | NDArray[np.integer]

ScalarTypeVar = TypeVar("ScalarTypeVar", bound=np.generic, covariant=True)


class ArrayBuffer(Generic[ScalarTypeVar]):
    """
    1. 支持双向 push/pop
    2. 支持随机访问
    3. 支持转换成 NDArray，支持直接当作一个 Array 进行各种运算
    """

    def __init__(
            self, dtype: DTypeLike = np.float64,
            init_shape: tuple[int] | int = (2048,),
            max_len: int | None = None,
    ):

        if isinstance(init_shape, int):
            init_shape = (init_shape,)
        self._dtype = np.dtype(dtype)
        self._init_shape = init_shape
        self._array: NDArray[ScalarTypeVar] = np.zeros(init_shape, dtype)
        self._start: int = 0
        self._stop: int = 0
        self._extensions = np.zeros(2048, dtype=[
            ('left', np.int32),
            ('right', np.int32),
            ('length', np.int32)
        ])
        self._extension_n: int = 0

        self._max_len = max_len

    @property
    def max_len(self) -> int | None:
        return self._max_len

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    @property
    def itemsize(self) -> int:
        return self._array.itemsize

    @property
    def shape(self) -> tuple[int, ...]:
        return len(self), *self._init_shape[1:]

    def __array__(self, *args: Any, **kwargs: Any) -> NDArray[ScalarTypeVar]:
        return np.asarray(self._array[self._start:self._stop, ...], *args, **kwargs)

    def __len__(self) -> int:
        return self._stop - self._start

    @overload
    def __getitem__(self, key: slice | tuple[IndexArrayType, ...]) -> NDArray:
        ...

    @overload
    def __getitem__(self, key: IndexArrayType | SupportsIndex) -> NDArray | ScalarType:
        ...

    def __getitem__(
            self,
            key: slice | SupportsIndex | NDArray | tuple[IndexArrayType, ...]
    ) -> NDArray | ScalarType:
        return self._array[self._start:self._stop][key]

    @overload
    def __setitem__(self, key: slice | tuple[IndexArrayType, ...], value: ArrayLike) -> None:
        ...

    @overload
    def __setitem__(self, key: IndexArrayType | SupportsIndex, value: ScalarType | ArrayLike) -> None:
        ...

    def __setitem__(
            self,
            key: slice | tuple[IndexArrayType, ...] | IndexArrayType | SupportsIndex,
            value: ArrayLike | ScalarType
    ) -> None:
        self._array[self._start:self._stop][key] = value

    def __iter__(self) -> Iterator[Any]:
        return iter(self._array[self._start:self._stop])

    def state_dict(self) -> dict:
        """
        For JSON serialization
        Returns:

        """
        return {
            "dtype": self.dtype.str,
            "data": np.asarray(self).tolist(),
            "max_len": self.max_len,
        }

    @staticmethod
    def load(state_dict: dict) -> 'ArrayBuffer':
        ret = ArrayBuffer.__new__(ArrayBuffer)
        ret.__setstate__(state_dict)
        return ret

    def __getstate__(self) -> dict:
        return {
            "dtype": self.dtype.str,
            "data": self.__array__().tolist(),
            "max_len": self.max_len,
        }

    def __setstate__(self, state: dict) -> None:
        ArrayBuffer.__init__(self, dtype=np.dtype(state['dtype']), max_len=state['max_len'])
        self.extend(state['data'])

    def _check_enlarge(self, side: Side, extension_length: int) -> None:

        if side != Side.NONE:
            self._extensions[self._extension_n] = (
                extension_length if side == Side.LEFT else 0,
                extension_length if side == Side.RIGHT else 0,
                len(self) + extension_length
            )
            self._extension_n = (self._extension_n + 1) % (len(self._extensions))

        if side == Side.RIGHT and len(self._array) - self._stop >= extension_length:
            return
        elif side == Side.LEFT and self._start >= extension_length:
            return

        left_ext = np.sum(self._extensions['left'])
        right_ext = np.sum(self._extensions['right'])
        max_length = np.max(self._extensions['length'])

        mult = np.int64(np.ceil(max_length / max(1, left_ext + right_ext)))

        left_ext *= mult
        right_ext *= mult

        new_shape = (
            left_ext + max(len(self), self._init_shape[0]) + extension_length + right_ext,
            *self._array.shape[1:]
        )

        if side == Side.LEFT:
            new_stop = new_shape[0] - right_ext
            new_start = new_stop - len(self)
        else:
            new_start = left_ext
            new_stop = new_start + len(self)

        assert new_stop - new_start == len(self)

        new_arr = np.zeros(new_shape, self._dtype)
        if new_start < new_stop:
            new_arr[new_start:new_stop] = self[:]

        self._start = new_start
        self._stop = new_stop
        self._array = new_arr

    def append(self, item: ScalarType) -> Self:
        return self.extend(np.asarray([item]))

    def appendleft(self, item: ScalarType) -> Self:
        return self.extendleft(np.asarray([item]))

    def extend(self, other: ArrayLike) -> Self:
        other = np.asarray(other)
        ol = len(other)

        self._check_enlarge(Side.RIGHT, ol)
        self._array[self._stop:self._stop + ol] = other[:]
        self._stop += ol
        if self.max_len is not None and len(self) > self.max_len:
            self.popleft(len(self) - self.max_len)
        return self

    def extendleft(self, other: ArrayLike) -> Self:
        other = np.asarray(other)

        ol = len(other)

        self._check_enlarge(Side.LEFT, ol)
        self._array[self._start - ol:self._start] = other[:]
        self._start -= ol
        if self.max_len is not None and len(self) > self.max_len:
            self.pop(len(self) - self.max_len)
        return self

    def pop(self, count: int) -> NDArray:

        count = min(count, len(self))

        ret: NDArray = self[-count:]

        self._stop -= count

        return ret

    def popleft(self, count: int) -> NDArray:

        count = min(count, len(self))

        ret: NDArray = self[:count]

        self._start += count

        return ret

    def clear(self) -> Self:

        self._start = 0
        self._stop = 0
        self._check_enlarge(Side.NONE, 0)
        return self

    def __add__(self, other: ArrayLike) -> NDArray[ScalarTypeVar]:
        return np.asarray(self.__array__() + np.asarray(other, dtype=self.dtype))

    def __iadd__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.add(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __sub__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() - np.asarray(other, dtype=self.dtype))

    def __isub__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.subtract(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __mul__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() * np.asarray(other, dtype=self.dtype))

    def __imul__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.multiply(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __truediv__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() / np.asarray(other, dtype=self.dtype))

    def __itruediv__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.true_divide(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __floordiv__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() // np.asarray(other, dtype=self.dtype))

    def __ifloordiv__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.floor_divide(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __lt__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() < np.asarray(other, dtype=self.dtype))

    def __gt__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() > np.asarray(other, dtype=self.dtype))

    def __le__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() <= np.asarray(other, dtype=self.dtype))

    def __ge__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() >= np.asarray(other, dtype=self.dtype))

    def __eq__(self, other: ArrayLike) -> NDArray[np.bool_]:  # type: ignore
        return np.asarray(self.__array__() == np.asarray(other, dtype=self.dtype))

    def __mod__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() % np.asarray(other, dtype=self.dtype))

    def __imod__(self, other: ArrayLike) -> Self:
        other_arr: NDArray[ScalarTypeVar] = np.asarray(other, dtype=self.dtype)
        self._array[self._start:self._stop, ...] = np.mod(self._array[self._start:self._stop, ...], other_arr)
        return self

    def __matmul__(self, other: ArrayLike) -> NDArray:
        return np.asarray(self.__array__() @ np.asarray(other, dtype=self.dtype))

    def __neg__(self) -> NDArray[ScalarTypeVar]:
        return -self.__array__()

    def __repr__(self) -> str:
        return f"ArrayBuffer(data={self.__array__()!r}, max_len={self.max_len})"
