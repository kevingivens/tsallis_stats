import numpy as np

from scipy._lib.doccer import extend_notes_in_docstring
import scipy.special as sc
from scipy.stats._distn_infrastructure import rv_continuous


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
    def q_exp(q, x):
        assert(q < 3)
        return np.where((1.0+(1.0-q)*x)**(1.0/(1.0-q)))

    @staticmethod
    def q_log(q, x):
        # assert(x > 0)
        assert(q < 3)
        return (1.0/(1.0-q))*(x**(1.0-q)-1.0)
    
    @staticmethod
    def _get_m(q):
        """ first moment """
        return np.where(q < 3, 0, np.nan)
    
    @staticmethod
    def _get_v(q, beta):
        """ second moment """
        if q < 5/3:
            result = 1/(beta * (5-3*q))
        elif q < 2:
            result = np.inf
        else:
            result = np.nan
        return result
    
    @staticmethod
    def _get_s(q):
        """ third moment """
        return np.where(q < 3/2, 0, np.nan)
    
    @staticmethod
    def _get_k(q):
        """ fourth moment """
        return np.where(q < 7/5, 6*(q-1)/(7-5*q), np.nan)
    
    def _argcheck(self, q, beta):
        return (q < 3) & (beta > 0)

    def _get_support(self, q, beta):
        _b = np.where(q > 1, 1.0 / np.sqrt(beta*(1- q)), np.inf)
        _a = np.where(q < 1, -1.0 / np.sqrt(beta*(1- q)), -np.inf)
        return _a, _b

    def _pdf(self, x, q, beta):
        if q<0:
            c_q = 2.0*np.sqrt(np.pi)*sc.gamma(1.0/(1.0-q))
            c_q /= (3.0-q)*np.sqrt(1.0-q)*sc.gamma((3.0-q)/(2.0*(1.0-q)))
        elif q==1:
            c_q = np.sqrt(np.pi)
        else:
            c_q = 2.0*np.sqrt(np.pi)*sc.gamma(1.0/(1.0-q))
            c_q /= (3.0-q)*np.sqrt(1.0-q)*sc.gamma((3.0-q)/(2.0*(1.0-q)))

        return np.sqrt(beta/c_q)*self.q_exp(-beta*x**2)


    def _rvs(self, q, beta, size=None, random_state=None):
        u1 = random_state.uniform(size=size)
        u2 = random_state.uniform(size=size)
        q_prime = (1 + q)/(3 - q)
        z = np.sqrt(-2.0*self.q_log(q_prime, u1)) * np.cos(2*np.pi*u2)
        return self._get_m(q) - z/(np.sqrt(beta*(3 - q)))
    
    #def _cdf(self, x, q, beta):
    #    pass

    #def _ppf(self, q, beta):
    #    pass

    def _stats(self, q, beta):
        m = self._get_m(q)
        v = self._get_v(q, beta)
        s = self._get_s(q)
        k = self._get_k(q)
        return m, v, s, k


q_gaussian = q_gaussian_gen(name='q_gaussian')