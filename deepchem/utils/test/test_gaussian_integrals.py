import pytest


@pytest.mark.torch
def test_boys():
    import mpmath
    from deepchem.utils.gaussian_integrals.integral_utils import boys

    # Test output length
    for mmax in [0, 1, 5, 10]:
        assert len(boys(mmax, 2.0)) == mmax + 1

    # Test F_0(0) -> 1 case
    assert abs(boys(0, 0)[0] - 1.0) < 1e-15

    # Test F_m(0) = 1 / (2m + 1) for all m
    result = boys(6, 0.0)
    for m in range(6 + 1):
        expected = 1.0 / (2 * m + 1)
        assert abs(float(result[m]) - expected) < 1e-15

    # Test F_0(x) -> sqrt(pi / 4x) as x -> large
    x = 100
    assert abs(boys(0, x)[0] - float(mpmath.sqrt(mpmath.pi / (4 * x)))) < 1e-10
