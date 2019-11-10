import matlab.engine
import config

eng = matlab.engine.start_matlab()
eng.addpath(config.FASTMMD_MATLAB_PATH, nargout=0)
eng.demo(nargout=0)
