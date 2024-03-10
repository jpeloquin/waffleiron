class SelectionException(Exception):
    """Exception for invalid selections."""

    pass


class InvalidConditionError(ValueError):
    """Raise when an otherwise valid value would break an invariant"""

    pass


class UnsupportedFormatError(ValueError):
    """Exception for reads or writes to unsupported file formats.

    Example of use:

    FEBio file format version 1.2 is not fully supported.  Attempts to
    read an FEBio file format 1.2 file `foo.feb` should raise

    UnsupportedFormat('foo.feb', '1.2')

    """

    def __init__(self, msg, pth, form=None):
        super(UnsupportedFormatError, self).__init__(msg)
        self.file_path = pth
        self.file_format = form
        self.message = msg
