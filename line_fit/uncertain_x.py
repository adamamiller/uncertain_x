import numpy as np
from scipy.optimize import minimize


def neg_lnlike_unc_abscissa(theta, x, y, x_unc, y_unc, rhoxy = 1):
    """
    Calculate the log-likelihood for line with uncertainties on x and y
    
    Parameters
    ----------
    theta : tuple
        Tuple with the "angle" and "perpendicular offset" of the projected line
    
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
    print(theta)
    th, bperp = theta
    lnl = 0.
    v = np.array([[-np.sin(th)], [np.cos(th)]])
    vT = v.transpose()
    if rhoxy == 1:
        rhoxy = np.ones_like(x_unc)
    for (xx, yy, sx, sy, rxy) in zip(x, y, x_unc, y_unc, rhoxy):
        S_matrix = np.array([[ sx**2, rxy*sx*sy],
                    [rxy*sx*sy, sy**2.]]) 
        Z_matrix = np.array([[xx],[yy]])
        Delta = vT @ Z_matrix - bperp
        Sigma2 = vT @ S_matrix @ v
        lnl -= Delta**2. / (2. * Sigma2)
    
    neg_lnl = -1*lnl
    return neg_lnl

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
        """
        
        if not hasattr(self, 'x_unc'):
            raise RuntimeError('''Must associate data with object
                                run add_data() first''')
        
        guess_0 = np.append([np.arctan(slope_guess)], 
                            [intercept_guess*np.cos(np.arctan(slope_guess))])
                   
        mle = minimize(neg_lnlike_unc_abscissa, guess_0, method='BFGS',
                       args=(self.x, self.y, self.x_unc, self.y_unc))
        
        th, bperp = mle.x
        
        self.m = np.tan(th)
        self.b = bperp/np.cos(th)