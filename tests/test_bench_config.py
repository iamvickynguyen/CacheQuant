from cachequant.bench.config import DEFAULT_CONFIG


def test_default_config_has_positive_price_and_threads():
    assert DEFAULT_CONFIG.dollars_per_hour > 0
    assert DEFAULT_CONFIG.cpu_threads > 0
    assert DEFAULT_CONFIG.price_source
    assert DEFAULT_CONFIG.instance_type


def test_default_config_has_max_cache_tokens():
    assert DEFAULT_CONFIG.max_cache_tokens == 2048
