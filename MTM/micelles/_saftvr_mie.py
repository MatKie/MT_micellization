from sgtpy import component


class SaftVR(object):
    def __init__(self):
        self.__water = component(
            "water",
            ms=1.7311,
            sigma=2.4539,
            eps=110.85,
            lambda_r=8.308,
            lambda_a=6.0,
            eAB=1991.07,
            rcAB=0.5624,
            rdAB=0.4,
            sites=[0, 2, 2],
            cii=1.5371939421515458e-20,
        )
        self.__PentaneComponent = component(
            "pentane", ms=2, sigma=4.248, eps=317.5, lambda_r=16.06, lambda_a=6.0
        )
        self.__HexaneComponent = component(
            "hexane", ms=2, sigma=4.508, eps=376.35, lambda_r=19.57, lambda_a=6.0
        )
        self.__HeptaneComponent = component(
            "heptane", ms=2, sigma=4.766, eps=436.13, lambda_r=23.81, lambda_a=6.0
        )
        self.__OctaneComponent = component(
            "octane",
            ms=3.0,
            sigma=4.227,
            eps=333.7,
            lambda_r=16.14,
            lambda_a=6.0,
            cii=5.8915398756858572e-19,
        )
        self.__NonaneComponent = component(
            "nonane", ms=3, sigma=4.406, eps=374.21, lambda_r=18.31, lambda_a=6.0
        )
        self.__DecaneComponent = component(
            "dodecane", ms=3, sigma=4.584, eps=415.19, lambda_r=20.92, lambda_a=6.0
        )
        self.__UndecaneComponent = component(
            "undecane", ms=4, sigma=4.216, eps=348.9, lambda_r=16.84, lambda_a=6.0
        )
        self.__components = [
            self.__PentaneComponent,
            self.__HexaneComponent,
            self.__HeptaneComponent,
            self.__OctaneComponent,
            self.__NonaneComponent,
            self.__DecaneComponent,
            self.__UndecaneComponent,
        ]
        # Sorry this is ugly, they should match the above sequence
        self.__kij_coeffs = [
            [-3.43303335e-06, 2.65261499e-03, -4.42357526e-01],
            [-3.61129302e-06, 2.76461939e-03, -4.71873413e-01],
            [-3.76062271e-06, 2.85952765e-03, -5.06227204e-01],
            [-3.76764999e-06, 2.88068999e-03, -4.70588225e-01],
            [-3.93111896e-06, 2.99157273e-03, -4.94123059e-01],
            [-4.09541375e-06, 3.10514750e-03, -5.22152479e-01],
            [-4.10217229e-06, 3.13499002e-03, -5.03836509e-01],
        ]
        bij_linear = [
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
            [-0.00140671, 0.90335974],
        ]
        bij_quadratic = [
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
            [3.13521348e-05, -2.05774939e-02, 3.82334770e00],
        ]
        self.__bij_coeffs = {"linear": bij_linear, "quadratic": bij_quadratic}
