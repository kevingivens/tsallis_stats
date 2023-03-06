import numpy as np
import numpy.typing as npt
from scipy._lib.doccer import extend_notes_in_docstring
import scipy.special as sc
from scipy.stats._distn_infrastructure import rv_continuous, _ShapeInfo
from scipy._lib._util import _lazyselect, _lazywhere


class q_gaussian_gen(rv_continuous):
    r"""A q-Gaussian random variable.

    %(before_notes)s
    
    Notes
    -----
    The probability density function for `q_gaussian` is:
    .. math::
        f(x, q, b) = \frac{\sqrt(b)}{C_q}e_q(-b*x^2)
    where 
    :math:`e_q(x)` is a q-exponential defined by
    .. math::
        e_q(x) = [1 + (1-q)*x]_+^{\frac{1}{1-q}}
    and :math:`C_q` is a normalization factor given by
    C_q = \frac{2*\sqrt(\pi)*\Gamma(\frac{1}{1-q})}
               {(3-q)\sqrt{1-q}\Gamma(\frac{3-q}{2(1-q)})} for -\inf < q < 1

    C_q = \sqrt{\pi} for q = 1

    C_q = \frac{\sqrt(\pi)*\Gamma(\frac{3-q}{2(1-q)})}
               {\sqrt{q-1}\Gamma(\frac{1}{(q-1)})} for 1< q < 3

    for :math:`0 <= x <= 1`, :math:`q < 3`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).
    `q_gaussian` takes :math:`q` and :math:`b` as shape parameters.
    
    %(after_notes)s

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Q-Gaussian_distribution

    .. [2] W. Thistleton, J.A. Marsh, K. Nelson and C. Tsallis, 
           "Generalized Box–Muller method for generating q-Gaussian 
           random deviates", IEEE Transactions on Information Theory 
           53, 4805 (2007)

    %(example)s
    """

    @staticmethod
    def q_exp(x: npt.ArrayLike, q: float) -> npt.ArrayLike:
        assert(q < 3)
        # lazyselect
        if q == 1:
            return np.exp(x)
        elif 1.0+(1.0-q)*x > 0:
            return (1.0+(1.0-q)*x)**(1.0/(1.0-q))
        elif 1.0+(1.0-q)*x <= 0:
            return 0.0

    @staticmethod
    def q_log(x: npt.ArrayLike, q) -> npt.ArrayLike:
        assert(q < 3)
        return _lazywhere(q == 1, np.log(x), (x**(q-1.)-1.)/(1.-q))
    
    def _shape_info(self):
        iq = _ShapeInfo("q", False, (-np.inf, 3), (False, False))
        ibeta = _ShapeInfo("beta", False, (0, np.inf), (False, False))
        return [iq, ibeta]
    
    #def _argcheck(self, q: float, beta: float):
    #    return (q < 3) & (beta > 0)

    def _get_support(self, q: float, beta: float) -> tuple[float, float]:
        _a = _lazywhere(q < 1, -1.0/np.sqrt(beta*(1-q)), -np.inf)
        _b = _lazywhere(q < 1, 1.0/np.sqrt(beta*(1-q)), np.inf)
        return _a, _b

    def _pdf(self, x, q: float, beta: float) -> npt.ArrayLike:
        assert(q < 3)
        # TODO: replace with lazyselect
        if q<1:
            c_q = 2.0*np.sqrt(np.pi)*sc.gamma(1.0/(1.0-q))
            c_q /= (3.0-q)*np.sqrt(1.0-q)*sc.gamma((3.0-q)/(2.0*(1.0-q)))
        elif q==1:
            c_q = np.sqrt(np.pi)
        elif 1 < q < 3:
            c_q = np.sqrt(np.pi)*sc.gamma((3.0-q)/(2.0*q-2.0))
            c_q /= np.sqrt(q-1.0)*sc.gamma(1.0/(q-1.0))
        else:
            c_q = np.nan

        return np.sqrt(beta/c_q)*self.q_exp(-beta*x**2, q)


    def _rvs(self, q:float, beta:float, size=None, random_state=None):
        u1 = random_state.uniform(size=size)
        u2 = random_state.uniform(size=size)
        q_prime = (1 + q)/(3 - q)
        z = np.sqrt(-2.0*self.q_log(u1, q_prime)) * np.cos(2*np.pi*u2)
        return self._stats(q)[0] + z/(np.sqrt(beta*(3 - q)))
    
    #def _cdf(self, x, q, beta):
    #    pass

    #def _ppf(self, q, beta):
    #    pass

    #def _logpdf(self, x, q, beta):
    #    pass

    def _stats(self, q: float, beta: float) -> tuple[float, float, float, float]:
        
        mu = _lazywhere(q < 3, 0, np.nan)
  
        # TODO replace with lazyselect
        if q < 5/3:
            mu2 = 1/(beta * (5-3*q))
        elif q < 2:
            mu2 = np.inf
        else:
            mu2 = np.nan

        g1 = _lazywhere(q < 3/2, 0, np.nan)
        g2 = _lazywhere(q < 7/5, 6*(q-1)/(7-5*q), np.nan)
        return mu, mu2, g1, g2


q_gaussian = q_gaussian_gen(name='q_gaussian')