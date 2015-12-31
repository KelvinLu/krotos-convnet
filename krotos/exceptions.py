class ErrorCallException(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        return repr(self.status)
