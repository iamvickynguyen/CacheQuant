def test_kernel_package_and_numba_importable():
    import cachequant.kernel
    import numba

    assert cachequant.kernel is not None
    assert numba.njit is not None
