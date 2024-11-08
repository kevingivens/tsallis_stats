import numpy as np
import numpy.typing as npt
from scipy._lib.doccer import extend_notes_in_docstring
from scipy._lib._util import _lazyselect, _lazywhere
from scipy.stats._distn_infrastructure import rv_continuous, _ShapeInfo
import scipy.special as sc



def q_exp(x: npt.ArrayLike, q: float) -> npt.ArrayLike:
    """ q exponential function (q generalization of an exponential function)

    Parameters:
    ----------
        x : npt.ArrayLike
            array_like
        q : float
            degree of non-extensivity

    Returns:
    --------
        y : npt.ArrayLike 
            The q logarithm of x, element-wise. This is a scalar if x is a scalar.

    """
    y = _lazyselect(
        [q == 1,
         (q != 1) & (1.0+(1.0-q)*x > 0),
         (q != 1) & (1.0+(1.0-q)*x <= 0)],
        [lambda x_, q_: np.exp(x_),
         lambda x_, q_: (1.0+(1.0-q_)*x_)**(1.0/(1.0-q_)),
         lambda x_, q_: 0.0
         ],
        (x, q))
    return y
    

def q_log(x: npt.ArrayLike, q: float) -> npt.ArrayLike:
    """q natural logarithm function (q generalization of an logarithm function)

     Parameters:
    ----------
        x : npt.ArrayLike
            array_like
        q : float
            degree of non-extensivity

    Returns:
    --------
        y : npt.ArrayLike 
            The q logarithm of x, element-wise. This is a scalar if x is a scalar.

    """
    return _lazywhere(q == 1,
                      [x, q],
                      lambda x_, q_: np.log(x_), 
                      f2 = lambda x_, q_: (x_**(1.-q_)-1.)/(1.-q_))


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
    
    def _shape_info(self):
        iq = _ShapeInfo("q", False, (-np.inf, 3), (False, False))
        ibeta = _ShapeInfo("beta", False, (0, np.inf), (False, False))
        return [iq, ibeta]
    
    def _argcheck(self, q: float, beta: float):
        return (q < 3) & (beta > 0)

    def _get_support(self, q: float, beta: float) -> tuple[float, float]:
        _a = _lazywhere(
            q < 1,
            [q, beta], 
            lambda q_, beta_: -1.0/np.sqrt(beta_*(1-q_)), 
            f2 = lambda q_, beta_: -np.inf
        )
        _b = _lazywhere(
            q < 1, 
            [q, beta], 
            lambda q_, beta_: 1.0/np.sqrt(beta_*(1-q_)), 
            f2 = lambda q_, beta_: np.inf
        )
        return _a, _b

    def _pdf(self, x, q: float, beta: float) -> npt.ArrayLike:
        
        conditions = [q < 1, q == 1, (1 < q) & (q < 3)] 
        
        def c1(q):
            c_q = 2.0*np.sqrt(np.pi)*sc.gamma(1.0/(1.0-q))
            c_q /= (3.0-q)*np.sqrt(1.0-q)*sc.gamma((3.0-q)/(2.0*(1.0-q)))
            return c_q

        def c2(q):
            c_q = np.sqrt(np.pi)*sc.gamma((3.0-q)/(2.0*q-2.0))
            c_q /= np.sqrt(q-1.0)*sc.gamma(1.0/(q-1.0))
            return c_q

        c_q = _lazyselect(
            conditions,
            [c1, lambda q_: np.sqrt(np.pi), c2],
            [q],
        )
        
        return (np.sqrt(beta)/c_q)*q_exp(-beta*x**2, q)


    def _rvs(self, q:float, beta:float, size=None, random_state=None):
        u1 = random_state.uniform(size=size)
        u2 = random_state.uniform(size=size)
        q_prime = (1 + q)/(3 - q)
        z = np.sqrt(-2.0*q_log(u1, q_prime)) * np.cos(2*np.pi*u2)
        return self._stats(q, beta)[0] + z/(np.sqrt(beta*(3 - q)))
    

    def _stats(self, q: float, beta: float) -> tuple[float, float, float, float]:
        
        # TODO: consider lazywhere
        mu = np.where(q < 3, 0, np.nan)

        condlist = [
            q < 5/3, 
            (5/3 <= q) & (q < 2),
            (2 <= q) & (q < 3),
        ]
        
        mu2 = _lazyselect(
            condlist, 
            [
                lambda q_, beta_: 1/(beta_ * (5-3*q_)), 
                lambda q_, beta_: np.inf,
                lambda q_, beta_: np.nan,
            ],
            [q, beta],
        )
        
        # TODO: consider lazywhere
        g1 = np.where(q < 3/2, 0, np.nan)
        g2 = np.where(q < 7/5, 6*(q-1)/(7-5*q), np.nan)
        
        return mu, mu2, g1, g2


q_gaussian = q_gaussian_gen(name='q_gaussian')