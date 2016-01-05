class ParametersError(TypeError):
    def __init__(self, msg):
        super(ParametersError, self).__init__(msg)
