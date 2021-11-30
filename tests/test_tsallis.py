import sys

sys.path.append("..")

import numpy as np
import pytest
from scipy import stats
from scipy.stats.tests.test_continuous_basic import (
    check_sample_meanvar_, 
    # cases_test_cont_basic
)

from tsallis.tsallis import q_gaussian

# @pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
@pytest.mark.parametrize('distname,arg', [(q_gaussian, (1.2, 3))])
@pytest.mark.parametrize('sn, n_fit_samples', [(500, 200)])
def test_cont_basic(distname, arg, sn, n_fit_samples):
    # this test skips slow distributions

    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'

    rng = np.random.RandomState(765456)
    rvs = distfn.rvs(size=sn, *arg, random_state=rng)
    sm = rvs.mean()
    sv = rvs.var()
    m, v = distfn.stats(*arg)

    check_sample_meanvar_(distfn, arg, m, v, sm, sv, sn, distname + 'sample mean test')