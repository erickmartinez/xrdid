class Error(Exception):
    """
    Base class for exceptions in this module
    """
    pass


class InvalidClassProperty(Error):
    """
    This class defines a class property error.
    This type of exception will be raised whenever the class contains invalid properties (e.g. when some of its
    properties have not been defined).
    """

    def __init__(self, property: str, value: object):
        self._message = 'The property \'{0}\' has the invalid value \'{1}\'.'.format(property, value)
        self._property: str = property
        self._value: object = value

    @property
    def message(self):
        return self._message

    @property
    def property(self):
        return '{0}: {1}'.format(self._property, self._value)
