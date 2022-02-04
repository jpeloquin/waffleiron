from waffleiron.febio import read_log
from waffleiron.test.fixtures import DIR_FIXTURES


def test_read_boxed_error():
    """Test if waffleiron can read boxed error messages

    Can we read an error message like the following one from an FEBio
    log file?

    *************************************************************************
    *                                ERROR                                  *
    *                                                                       *
    * Model initialization failed                                           *
    *                                                                       *
    *************************************************************************

    """
    p = DIR_FIXTURES / "log_boxed_error.log"
    errors = read_log(p)
    assert len(errors) == 1
    assert errors[0] == "Model initialization failed"
