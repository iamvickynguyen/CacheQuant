def test_cachequant_importable():
    import cachequant
    import cachequant.bench

    assert cachequant is not None
    assert cachequant.bench is not None
