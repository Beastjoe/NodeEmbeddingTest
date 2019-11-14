from graspy.inference import LatentDistributionTest

def ase_baseline_test(source_ad, target_ad):
    """

    :param source_ad: 2d ndarray - Adjacency Matrix of source
    :param target_ad: 2d ndarray - Adjacency Matrix of target
    :return:
    """
    ldt = LatentDistributionTest()
    p = ldt.fit(source_ad, target_ad)
    print("P value: " + str(p))
    return p