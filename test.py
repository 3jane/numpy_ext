import pytest
import numpy as np
import pandas as pd

import numpy_ext as npext


@pytest.fixture(params=[1, 2, -1])
def n_jobs(request):
    return request.param


@pytest.fixture(params=[2, 3, 4, 5, 6])
def expanding_window(request):
    return request.param


@pytest.fixture(params=[np.max, np.min, np.sum])
def expanding_func(request):
    return request.param


@pytest.fixture(params=[npext.rolling_apply, npext.expanding_apply])
def apply(request):
    return request.param


@pytest.mark.parametrize(
    'input_arr, res_arr',
    [
        (np.array([1, np.nan, 3]), np.array([1., 3.])),
        (np.array([]), np.array([])),
        (np.array([3, 0]), np.array([3, 0]))
    ]
)
def test_dropna(input_arr, res_arr):
    assert np.array_equal(
        npext.drop_na(input_arr), res_arr
    )


@pytest.mark.parametrize(
    'input_arr, value, res_arr',
    [
        (np.array([1, 2, 3]), 0, np.array([1, 2, 3])),
        (np.array([1, 2, np.nan, 3]), 0, np.array([1, 2, 0, 3])),
        (np.array([1, 2, np.nan, 3, np.inf]), 0, np.array([1, 2, 0, 3, np.inf])),

    ]
)
def test_fillna(input_arr, value, res_arr):
    assert np.array_equal(
        npext.fill_na(input_arr, value), res_arr
    )


@pytest.mark.parametrize(
    'input_arr, value, res_arr',
    [
        (np.array([1, 2, 3]), 0, np.array([1, 2, 3])),
        (np.array([1, 2, np.nan, 3]), 0, np.array([1, 2, 0, 3])),
        (np.array([1, 2, np.nan, 3, np.inf]), 0, np.array([1, 2, 0, 3, 0])),

    ]
)
def test_fillnotfinite(input_arr, value, res_arr):
    assert np.array_equal(
        npext.fill_not_finite(input_arr, value), res_arr
    )


def test_prepandna():
    array = np.array([1, 2.5, 0, 1, 3])
    res = npext.prepend_na(array, 5)

    assert res.size == array.size + 5
    assert np.isnan(res[:5]).all()
    assert np.array_equal(res[5:], array)


@pytest.mark.parametrize(
    'array, test_res, null_check_func',
    [
        (
            np.array([1, 2.5, 0, 1, 3]),
            np.array([
                [np.nan, np.nan, 1],
                [np.nan, 1., 2.5],
                [1., 2.5, 0.],
                [2.5, 0., 1.],
                [0., 1., 3.]
            ]),
            np.isnan
        ),
        (
            np.array(['2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13'], dtype='datetime64[D]'),
            np.array([
                ['NaT', 'NaT', '2020-04-09'],
                ['NaT', '2020-04-09', '2020-04-10'],
                ['2020-04-09', '2020-04-10', '2020-04-11'],
                ['2020-04-10', '2020-04-11', '2020-04-12'],
                ['2020-04-11', '2020-04-12', '2020-04-13']], dtype='datetime64[D]'),
            np.isnat
        )
    ]
)
def test_rolling(array, test_res, null_check_func):
    res = npext.rolling(array, 3, as_array=True)
    for i in range(test_res.shape[0]):
        for j in range(test_res.shape[1]):
            v, test_v = res[i, j], test_res[i, j]
            assert null_check_func(v) and null_check_func(test_v) or v == test_v


def test_rolling_gen():
    array = np.array([1, 2.5, 0, 1, 3])
    res = np.array([
        [np.nan, np.nan, 1],
        [np.nan, 1, 2.5],
        [1., 2.5, 0.],
        [2.5, 0., 1.],
        [0., 1., 3.]
    ])
    res_gen = np.array(list(npext.rolling(array, 3)))
    res_gen_skip = np.array(list(npext.rolling(array, 3, skip_nans=True)))

    assert np.isnan(res_gen[:2]).any()
    assert np.array_equal(res_gen[2:], res[2:])
    assert np.array_equal(res_gen_skip, res[2:])


@pytest.mark.parametrize(
    'apply, input, window, func, params, test_result',
    [
        (
            npext.rolling_apply,
            [
                np.array([1, 2.5, 0, 1, 3]),
                np.array([1, 2.5, 0, 1, 3]),
                np.array([1, 2.5, 0, 1, 3])
            ],
            3,
            lambda x, y, z, v: sum([sum(x), sum(y), sum(z)]) + v,
            dict(v=1),
            np.array([np.nan, np.nan, 11.5, 11.5, 13])
        ),
        (
            npext.rolling_apply,
            np.array([1, 2.5, 0, 1, 3]),
            3,
            sum,
            {},
            np.array([np.nan, np.nan, 3.5, 3.5, 4])
        ),
        (
            npext.expanding_apply,
            [
                np.array([1, 2.5, 0, 1, 3]),
                np.array([1, 2.5, 0, 1, 3]),
                np.array([1, 2.5, 0, 1, 3])
            ],
            3,
            lambda x, y, z, v: sum([sum(x), sum(y), sum(z)]) + v,
            dict(v=1),
            np.array([np.nan, np.nan, 11.5, 14.5, 23.5])
        ),
        (
            npext.expanding_apply,
            np.array([1, 2.5, 0, 1, 3]),
            3,
            sum,
            {},
            np.array([np.nan, np.nan, 3.5, 4.5, 7.5])
        )
    ]
)
def test_window_apply_func(apply, input, window, func, params, test_result, n_jobs):
    if isinstance(input, np.ndarray):
        res = apply(
            func, window, input, n_jobs=n_jobs, **params
        )
    else:
        res = apply(
            func, window, *input, n_jobs=n_jobs, **params
        )

    assert test_result.size == res.size
    assert np.isnan(res[:window - 1]).all()
    assert np.array_equal(
        res[window - 1:],
        test_result[window - 1:]
    )


@pytest.mark.parametrize(
    'input',
    [
        [np.zeros((3, 3))],
        [
            np.arange(8),
            np.arange(9)
        ]
    ]
)
def test_apply_input_errors(input, apply):
    with pytest.raises(ValueError):
        apply(lambda *args: 1, 2, *input)


@pytest.mark.parametrize(
    'window',
    [None, 2.]
)
def test_rolling_apply_wrong_window_type(window, apply):
    with pytest.raises(TypeError):
        apply(sum, window, np.arange(20))


def test_nans_array():
    arr = npext.nans(5)
    assert arr.size == 5
    assert np.isnan(arr).all()


@pytest.mark.parametrize(
    'params, test_result',
    [
        (
            dict(start=-1, end=-100, min_step=1, step_mult=1.5),
            np.array([-1.0, -2.0, -3.5, -5.75, -9.125, -14.1875, -21.78125, -33.171875, -50.2578125, -75.88671875])
        ),
        (
            dict(start=1, end=100, min_step=1, step_mult=1.5, round_func=lambda a: a.astype(np.int)),
            np.array([1, 2, 3, 5, 9, 14, 21, 33, 50, 75])
        ),
        (
            dict(start=1, end=10, min_step=0.5, step_mult=1.5),
            np.array([1.0, 1.5, 2.25, 3.375, 5.0625, 7.59375])
        ),
        (
            dict(start=-100, end=-200, min_step=1, step_mult=1.25, round_func=lambda arr: [round(x, 2) for x in arr]),
            np.array([-100.0, -101.0, -102.25, -103.81, -105.77,
                      -108.21, -111.26, -115.07, -119.84, -125.8,
                      -133.25, -142.57, -154.21, -168.76, -186.95])
        ),
    ]
)
def test_expstep_range(params, test_result):
    assert np.array_equal(npext.expstep_range(**params), test_result)


@pytest.mark.parametrize(
    'params, error_msg',
    [
        (
            dict(start=-100, end=-200, min_step=0, step_mult=1.25),
            'min_step should be bigger than 0'
        ),
        (
            dict(start=-100, end=-200, min_step=-1, step_mult=1.25),
            'min_step should be bigger than 0'
        ),
        (
            dict(start=-100, end=-200, min_step=1, step_mult=0),
            'mult_step should be bigger than 0'
        ),
        (
            dict(start=-100, end=-200, min_step=1, step_mult=-1),
            'mult_step should be bigger than 0'
        ),
    ]
)
def test_expstep_range_wrong_params(params, error_msg):
    with pytest.raises(ValueError) as err:
        npext.expstep_range(**params)
    err.match(error_msg)


def test_expanding_pandas_similar(expanding_window, expanding_func, n_jobs):
    array = np.array([1, 2, 3, 5, -1, -3, 0.5, 20])
    res = npext.expanding_apply(expanding_func, expanding_window, array, n_jobs=n_jobs)
    test_res = pd.DataFrame(dict(x=array)).expanding(expanding_window).apply(expanding_func, raw=False).x.values

    nans_offset = np.isnan(test_res).sum()
    assert np.isnan(res[:nans_offset]).all()

    assert np.array_equal(
        test_res[nans_offset:],
        res[nans_offset:]
    )


@pytest.mark.parametrize(
    'arr, res_array',
    [
        ([[2, 2], [3, 3]], np.array([[0, 0], [1, 1]])),
        (np.array([[2, 2], [3, 3]]), np.array([[0, 0], [1, 1]])),
        ([1, 2, 3], np.array([0, 0, 1])),
        (
            [
                [[2, 2], [3, 3]],
                [[3, 3], [2, 2]],
            ],
            np.array([
                [[0, 0], [1, 1]],
                [[1, 1], [0, 0]],
            ])
        )
    ]
)
def test_apply(arr, res_array):
    assert np.array_equal(
        npext.apply_map(lambda x: 0 if x < 3 else 1, arr),
        res_array
    )


@pytest.mark.parametrize(
    'func, params, exc_class, exc_message_pattern',
    [
        (
            npext.expanding,
            dict(array=np.arange(10), min_periods=11),
            ValueError,
            'array.size should be bigger than min_periods'
        ),
        (
            npext.rolling,
            dict(array=np.arange(10), window=11),
            ValueError,
            'array.size should be bigger than window'
        ),
        (
            npext.expanding,
            dict(array=np.arange(10), min_periods=11.),
            TypeError,
            "Wrong min_periods"
        ),
        (
            npext.rolling,
            dict(array=np.arange(10), window=11.),
            TypeError,
            "Wrong window"
        ),

    ]
)
def test_window_func_exceptions(func, params, exc_class, exc_message_pattern):
    with pytest.raises(exc_class, match=exc_message_pattern):
        func(**params)


def test_expanding_with_nans():
    array = np.arange(10)
    res = npext.expanding(array, min_periods=2, skip_nans=False, as_array=True)

    assert len(array) == len(res)
    assert np.isnan(res[0]).all()
