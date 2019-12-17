import numpy as np
from scipy.optimize import minimize
import emcee


def lnlike_unc_abscissa(theta, x, y, x_unc, y_unc, rhoxy = 0):
    """
    Calculate the log-likelihood for line with uncertainties on x and y
    
    Parameters
    ----------
    theta : tuple
        Tuple with the "cosine angle" and "perpendicular offset" of 
        the projected line
    
    x : array-like
        Measurements for the coordinate projected along the abscissa
    
    y : array-like
        Measurements for the coordinate projected along the ordinate
    
    x_unc : array-like
        Estimate of the uncertainty for the abscissa coordinate. Note - 
        the uncertainty is assumed to be Gaussian
    
    y_unc : array-like
        Estimate of the uncertainty for the ordinate coordinate. Note - 
        the uncertainty is assumed to be Gaussian
    
    Returns
    -------
    neg_lnl : float
        The log-likelihood of the data given the model parameters
    """

    cos_th, bperp = theta
    lnl = 0.
    v = np.array([[-np.sin(np.arccos(cos_th))], [cos_th]])
    vT = v.transpose()
    if rhoxy == 0:
        rhoxy = np.zeros_like(x_unc)
    for (xx, yy, sx, sy, rxy) in zip(x, y, x_unc, y_unc, rhoxy):
        S_matrix = np.array([[ sx**2, rxy*sx*sy],
                    [rxy*sx*sy, sy**2.]]) 
        Z_matrix = np.array([[xx],[yy]])
        Delta = vT @ Z_matrix - bperp
        Sigma2 = vT @ S_matrix @ v
        lnl -= Delta**2. / (2. * Sigma2)
    
    return lnl

def neg_lnlike_unc_abscissa(theta, x, y, x_unc, y_unc, rhoxy=0):
    lnl = lnlike_unc_abscissa(theta, x, y, x_unc, y_unc, rhoxy=rhoxy)
    return -1*lnl
    
class UncertainAbscissa():
    '''Object with results of fit including uncertainties on the x-axis
    '''

    def __init__(self, name):
        self.name = name

    def add_data(self, x, y, x_unc, y_unc):
        """
        Store data as object attributes
        
        Attributes
        ----------
        x : array-like
            Measurements for the coordinate projected along the abscissa
        
        y : array-like
            Measurements for the coordinate projected along the ordinate
        
        x_unc : array-like
            Estimate of the uncertainty for the abscissa coordinate. Note - 
            the uncertainty is assumed to be Gaussian
        
        y_unc : array-like
            Estimate of the uncertainty for the ordinate coordinate. Note - 
            the uncertainty is assumed to be Gaussian
        """
        
        if len(x) != len(y) or len(x) != len(x_unc) or len(x) != len(y_unc):
            err_str = 'x, y (and their uncertainties) must be the same length'
            raise RuntimeError(err_str)
        
        self.x = x
        self.y = y
        self.x_unc = x_unc
        self.y_unc = y_unc

    def mle_slope_intercept(self, slope_guess=1.0, intercept_guess=0.0):
        """
        Estimate the slope and uncertainty of a line describing y vs. x
        
        Parameters
        ----------
        slope_guess : float (default = 1.0)
            Initial guess at the slope of the line
        
        intercept_guess : float (default = 0.0)
            Initial guess at the intercept of the line
        
        Attributes
        ----------
        m : float
            Maximum-likelihood estimate of the slope of the line
        
        b : float
            Maximum-likelihood estimate of the intercept of the line
        
        m_unc : float
            Maximum-likelihood estimate of the uncertainty on the slope

        b_unc : float
            Maximum-likelihood estimate of the uncertainty on the intercept
        """
        
        if not hasattr(self, 'x_unc'):
            raise RuntimeError('''Must associate data with object
                                run add_data() first''')
        
        guess_0 = np.append([np.cos(np.arctan(slope_guess))], 
                            [intercept_guess*np.cos(np.arctan(slope_guess))])
                   
        mle = minimize(neg_lnlike_unc_abscissa, guess_0, method='BFGS',
                       args=(self.x, self.y, self.x_unc, self.y_unc))
        
        cos_th, bperp = mle.x
        cos_th_unc, bperp_unc = np.sqrt(np.diag(mle.hess_inv))
        th = np.arccos(cos_th)
        th_unc = np.sin(th)/np.sqrt(1 - cos_th**2)*cos_th_unc
        cov = mle.hess_inv[0,1]
                
        self.m = np.tan(th)
        self.m_unc = th_unc/cos_th**2
        self.b = bperp/cos_th
        self.b_unc = self.b*np.sqrt((bperp_unc/bperp)**2 +
                                    (cos_th_unc/cos_th)**2 - 
                                    2*cov/(bperp*cos_th))
    
    def mcmc_slope_intercept(self, slope_guess=1.0, intercept_guess=0.0, 
                             nwalkers=40):
        """
        Estimate the slope and uncertainty of a line describing y vs. x
        
        Parameters
        ----------
        slope_guess : float (default = 1.0)
            Initial guess at the slope of the line
        
        intercept_guess : float (default = 0.0)
            Initial guess at the intercept of the line
        
        nwalkers : int (default = 40)
            Number of walkers in the affine-invariant sampler
        
        Attributes
        ----------
        m_mcmc : float
            MCMC estimate of the median posterior slope
        
        b_mcmc : float
            MCMC estimate of the median intercept slope
        
        m_mcmc_unc : float
            Half the 84 - 16 percentile width of the posterior slope

        b_mcmc_unc : float
            Half the 84 - 16 percentile width of the posterior intercept
        """
        
        if not hasattr(self, 'x_unc'):
            raise RuntimeError('''Must associate data with object
                                run add_data() first''')
        
        guess_0 = np.append([np.arctan(slope_guess)], 
                            [intercept_guess*np.cos(np.arctan(slope_guess))])
        
        ndim = len(guess_0)
        nfac = [10**(-2.5)]*ndim

        #initial position of walkers
        rand_pos = [1 + nfac*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        lnlike_unc_abscissa, 
                                        args=(self.x, self.y, 
                                              self.x_unc, self.y_unc))
        old_tau = np.inf
        for sample in sampler.sample(rand_pos, 
                                     iterations=5e3, 
                                     progress=True):
            if sampler.iteration % int(5e2):
                continue
        
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
                    
        print(np.max(tau))
        samples = sampler.get_chain(discard=int(np.max(tau)), 
                                    # thin=np.max([int(np.max(tau)), 1]),
                                    flat=True)
        th_samp = samples[:,0]
        bperp_samp = samples[:,1]
        
        self.m_mcmc = np.percentile(np.tan(th_samp), 50)
        self.m_mcmc_unc = np.diff(np.percentile(np.tan(th_samp), 
                                  (15.865, 84.135)))
        self.b_mcmc = np.percentile(bperp_samp/np.cos(th_samp), 50)
        self.b_mcmc_unc = np.diff(np.percentile(bperp_samp/np.cos(th_samp), 
                                                (15.865, 84.135)))