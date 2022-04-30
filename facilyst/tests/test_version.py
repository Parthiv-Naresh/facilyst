from facilyst import __version__


def test_version():
    assert __version__ == "0.0.2"


def test_strs():
    this = "is a str with {name}, and {name} numbers"
    print(this.format(name="3"))
