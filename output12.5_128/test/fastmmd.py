import matlab
import config
import random
import math


def mmd_test(source, target, allSigmoid, nBasis, sample_size, method, threshold=0.05):
    """

    :param source: two sample sets
    :param target: two sample sets
    :param allSigmoid: bandwidth parameter for Gaussian kernel, scale or vector
    :param nBasis: number of basis for approximating p(w)
    :param sample_size: sample size
    :param method: test method
    :param threshold: 0.05
    :return:
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(config.FASTMMD_MATLAB_PATH, nargout=0)
    source_sample = matlab.double(source)
    target_sample = matlab.double(target)
    allSigmoid = matlab.double(allSigmoid)
    if method == "MMD-unbiased":
        d1, f1, ds1, ds2, ds3 = eng.MMD3(source_sample, target_sample, allSigmoid, nargout=5)
        print(ds2)
    elif method == "MMD-biased":
        d1, f1 = eng.MMD3(source_sample, target_sample, allSigmoid, nargout=2)
        print(d1)
    elif method == "FastMMD-Fourier":
        d2, f2 = eng.MMDFourierFeature(source_sample, target_sample, allSigmoid, nBasis, nargout=2)
        print("Biased:")
        print(d2)
        print("Unbiased:")
        print(f2)
    elif method == "FastMMD-Fastfood":
        d3, f3 = eng.MMDFastfood(source_sample, target_sample, allSigmoid, nBasis, nargout=2)
        print("Biased:")
        print(d3)
        print("Unbiased:")
        print(f3)
    elif method == "MMD-linear":
        f4 = eng.MMDlinear(source_sample, target_sample, allSigmoid, nargout=1)
        print(f4)


def _test_threshold(score, threshold, sample_size, biased=False):
    if biased:
        if score <= math.sqrt(2 / sample_size) * (1 + math.sqrt(2 * math.log(1 / threshold, math.e))):
            print("Pass")
            return True
        else:
            print("Fail")
            return False
    else:
        if score * score <= 4 / math.sqrt(sample_size) * math.sqrt(math.log(1 / threshold, math.e)):
            print("Pass")
            return True
        else:
            print("Fail")
            return False