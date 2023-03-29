import numpy as np
import pytest
from scipy import stats
from scipy.stats.tests.test_continuous_basic import (
    check_sample_meanvar_,
    check_cdf_ppf,
    check_sf_isf,
    check_pdf,
    check_pdf_logpdf,
    check_pdf_logpdf_at_endpoints,
    check_cdf_logcdf,
    check_sf_logsf,
    check_ppf_broadcast
    # cases_test_cont_basic
)

from tsallis_stats.tsallis import q_gaussian

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

    check_sample_meanvar_(m, v, rvs)
    check_cdf_ppf(distfn, arg, distname)
    check_sf_isf(distfn, arg, distname)
    check_pdf(distfn, arg, distname)
    check_pdf_logpdf(distfn, arg, distname)
    check_pdf_logpdf_at_endpoints(distfn, arg, distname)
    check_cdf_logcdf(distfn, arg, distname)
    check_sf_logsf(distfn, arg, distname)
    check_ppf_broadcast(distfn, arg, distname)


