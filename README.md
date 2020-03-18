# NumPy Extensions
[![Build Status - GitHub](https://github.com/3jane/numpy_ext/workflows/Python%20package/badge.svg)](https://github.com/3jane/numpy_ext/actions?query=workflow%3A%22Python+package%22) [![Build Status - GitHub](https://github.com/3jane/numpy_ext/workflows/Deploy%20docs/badge.svg)](https://github.com/3jane/numpy_ext/actions?query=workflow%3A%22Deploy+docs%22) ![Deploy PYPI](https://github.com/3jane/numpy_ext/workflows/Deploy%20PYPI/badge.svg)  [![Coverage Status](https://coveralls.io/repos/github/3jane/numpy_ext/badge.svg)](https://coveralls.io/github/3jane/numpy_ext)

An extension library for [NumPy](https://github.com/numpy/numpy) that implements common array operations not present in NumPy.

* `npext.fill_na(...)`
* `npext.drop_na(...)`
* `npext.rolling(...)`
* `npext.expanding(...)`
* `npext.rolling_apply(...)`
* `npext.expanding_apply(...)`
* `# etc`

---
## Documentation

* [API Reference](http://3jane.github.io/numpy_ext/)

## Installation
Regular installation:
```bash
pip install numpy_ext
```

For development:
```bash
git clone https://github.com/3jane/numpy_ext.git
cd numpy_ext
pip install -e .[dev]  # note: make sure you are using pip>=20
```

## Examples
Here are few common examples of how the library is used. The rest is available in the [documentation](http://3jane.github.io/numpy_ext/).

1) Apply a function to a rolling window over the provided array

```python
import numpy as np
import numpy_ext as npext

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
window = 3

npext.rolling_apply(np.sum, window, a)

> array([nan, nan,  3.,  6.,  9., 12., 15., 18., 21., 24.])
```

2) Same as the above, but with a custom function, two input arrays and parallel computation using `joblib`:

```python
def func(array_first, array_second, param):
    return (np.min(array_first) + np.sum(array_second)) * param


a = np.array([0, 1, 2, 3])
b = np.array([3, 2, 1, 0])

npext.rolling_apply(func, 2, a, b, n_jobs=2, param=-1)

> array([nan, -5., -4., -3.])
```

3) Same as the first example, but using **rolling** function:

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
window = 3

rolls = npext.rolling(a, window, as_array=True)

np.sum(rolls, axis=1)

> array([nan, nan,  3.,  6.,  9., 12., 15., 18., 21., 24.])
```

## License
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://tldrlegal.com/license/mit-license)

The software is distributed under MIT license. 
