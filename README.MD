[![Coverage badge](https://raw.githubusercontent.com/lizeyan/ndarraybuffer/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lizeyan/ndarraybuffer/blob/python-coverage-comment-action-data/htmlcov/index.html)

## Introduction
Wrap `numpy.ndarray` to support fast both-side insertion/deletion and size limit.

- Fully typed.
- Directly perform arithmetic operations with NDArray
- pickleable
- JSON serializable

```bash
pip install ndarraybuffer
```

### Both-side insertion/deletion
```python
arr = ArrayBuffer()
arr.extend(np.arange(100))
arr.append(100)
arr.appendleft(0)
arr.pop(1)
arr.popleft(1)
```

### Size limit
```python
arr = ArrayBuffer(max_len=10)
arr.extend(np.arange(10))
assert len(arr) == 10
assert np.array_equal(arr, np.arange(10))
arr.append(10)
assert len(arr) == 10
assert np.array_equal(arr, np.arange(1, 11))
```


### JSON Serialization and Deserialization
```python
arr = ArrayBuffer(max_len=10, dtype=np.float64)
arr.extend(np.arange(10))
state_dict = arr.state_dict()
rec = ArrayBuffer.load(json.loads(json.dumps(state_dict)))
```
