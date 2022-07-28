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
            "hexane",
            ms=2,
            sigma=4.508,
            eps=376.35,
            lambda_r=19.57,
            lambda_a=6.0,
            cii=3.7489834105357965e-19,
            Mw=86.18,
        )
        self.__HeptaneComponent = component(
            "heptane",
            ms=2,
            sigma=4.766,
            eps=436.13,
            lambda_r=23.81,
            lambda_a=6.0,
            cii=4.7586094841143767e-19,
            Mw=100.21,
        )
        self.__OctaneComponent = component(
            "octane",
            ms=3.0,
            sigma=4.227,
            eps=333.7,
            lambda_r=16.14,
            lambda_a=6.0,
            cii=5.9276345987710738e-19,
            Mw=114.13,
        )
        self.__NonaneComponent = component(
            "nonane",
            ms=3,
            sigma=4.406,
            eps=374.21,
            lambda_r=18.31,
            lambda_a=6.0,
            cii=7.3917359643968676e-19,
            Mw=128.2,
        )
        self.__DecaneComponent = component(
            "dodecane",
            ms=3,
            sigma=4.584,
            eps=415.19,
            lambda_r=20.92,
            lambda_a=6.0,
            cii=8.8898800228953004e-19,
            Mw=142.29,
        )
        self.__UndecaneComponent = component(
            "undecane",
            ms=4,
            sigma=4.216,
            eps=348.9,
            lambda_r=16.84,
            lambda_a=6.0,
            cii=1.0469158312483699e-18,
            Mw=156.31,
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
        # The pentane values are the same as for hexane. They are NOT
        # fitted to the pentane correlation, since there was no data from
        # the sources used. So just copied hexane values to pentane for
        # convenience.
        self.__bij_coeffs = [
            [7.26432341e-06, -5.43222205e-03, 1.46125112e00],
            [7.26432341e-06, -5.43222205e-03, 1.46125112e00],
            [1.41129869e-05, -9.64731035e-03, 2.09494088e00],
            [1.01385065e-05, -7.31757041e-03, 1.75636346e00],
            [6.93232444e-06, -5.40016085e-03, 1.45340261e00],
            [1.68755581e-05, -1.13018483e-02, 2.31617862e00],
            [1.62661849e-05, -1.10122365e-02, 2.27917150e00],
        ]
