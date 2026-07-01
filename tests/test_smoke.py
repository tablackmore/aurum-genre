import aurum_genre

def test_package_imports_and_has_version():
    assert isinstance(aurum_genre.__version__, str)
    assert aurum_genre.__version__.count(".") >= 1
