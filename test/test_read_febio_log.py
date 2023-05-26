from waffleiron.febio import LogFile
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
    log = LogFile(p)
    assert len(log.errors) == 1
    assert log.errors[0] == "Model initialization failed"
