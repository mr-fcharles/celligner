import pandas as pd
import os

#add to path /home/fcarli/francisCelligner/celligner
import sys
sys.path.append('/home/fcarli/francisCelligner/celligner')
sys.path.append('/home/fcarli/francisCelligner/celligner/mnnpy')

import numpy as np
from scipy import stats, special
import statsmodels.api as sm

from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

def squeezeVar(var, df):
    """
    Perform variance shrinkage using an empirical Bayes method.

    This function estimates a prior distribution for the variances and
    combines it with the observed variances to produce posterior estimates.

    Parameters:
    var (array-like): Observed variances
    df (int, float, or array-like): Degrees of freedom for each variance

    Returns:
    dict: A dictionary containing:
        - 'var_post': Posterior variance estimates
        - 'var_prior': Estimated prior variance
        - 'df_prior': Estimated prior degrees of freedom

    Raises:
    ValueError: If var is empty, or if lengths of var and df differ when df is array-like
    """
    n = len(var)
    
    # Check for empty input
    if n == 0:
        raise ValueError("var is empty")
    
    # Special case: single variance
    if n == 1:
        return {'var_post': var, 'var_prior': var, 'df_prior': 0}
    
    # Ensure df is an array of the same length as var
    if isinstance(df, (int, float)):
        df = np.full(n, df)
    else:
        if len(df) != n:
            raise ValueError("lengths differ")
    
    # Fit F-distribution to estimate prior
    out = fitFDist(var, df1=df)
    
    # Check if prior estimation was successful
    if out['df2'] is None or np.isnan(out['df2']):
        raise ValueError("Could not estimate prior df")
    
    # Rename output keys for clarity
    out['var_prior'] = out['scale']
    out['df_prior'] = out['df2']
    del out['df2'], out['scale']
    
    # Calculate total degrees of freedom
    df_total = df + out['df_prior']
    
    # Calculate posterior variances
    if np.isinf(out['df_prior']):
        # If prior df is infinite, posterior is equal to prior
        out['var_post'] = np.full(n, out['var_prior'])
    else:
        var = np.array(var)
        var[df == 0] = 0  # Guard against missing or infinite values
        # Weighted average of observed and prior variances
        out['var_post'] = (df * var + out['df_prior'] * out['var_prior']) / df_total
    
    return out



def fitFDist(x, df1):
    """
    Fit an F-distribution to the given data.

    This function estimates the parameters of an F-distribution based on the input data
    and degrees of freedom. It's particularly useful in the context of empirical Bayes
    methods for variance estimation.

    Parameters:
    x (array-like): The input data, typically variances or squared standard deviations.
    df1 (array-like): Degrees of freedom associated with each element in x.

    Returns:
    dict: A dictionary containing:
        'scale': The estimated scale parameter of the F-distribution.
        'df2': The estimated second degree of freedom parameter.

    Raises:
    Warning: If more than half of the variances are zero, or if zero variances are detected.
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    df1 = np.asarray(df1)

    # Filter out invalid values
    valid = np.isfinite(x) & np.isfinite(df1) & (x >= 0) & (df1 > 0)
    x = x[valid]
    df1 = df1[valid]
    n = len(x)

    # Return NaN if no valid data
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}

    # Calculate median and handle zero values
    m = np.median(x)
    if m == 0:
        print("Warning: More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1
    else:
        if np.any(x == 0):
            print("Warning: Zero sample variances detected, have been offset")

    # Offset very small values to avoid log(0)
    x = np.maximum(x, 1e-5 * m)

    # Calculate log values and adjust
    z = np.log(x)
    e = z - special.digamma(df1 / 2) + np.log(df1 / 2)
    emean = np.mean(e)

    # Estimate variance
    evar = np.mean(n / (n - 1) * (e - emean) ** 2 - special.polygamma(1, df1 / 2))

    # Calculate scale and df2 based on estimated variance
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        s20 = np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        s20 = np.exp(emean)

    return {'scale': s20, 'df2': df2}

def trigammaInverse(x):
    """
    Optimized version of the trigammaInverse function.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([])

    # Initialize y with NaNs
    y = np.full_like(x, np.nan)

    # Mask for valid entries: x not NaN and x >= 0
    valid_mask = (~np.isnan(x)) & (x >= 0)
    if not np.any(valid_mask):
        if np.any(~valid_mask):
            print("Warning: NaNs produced due to invalid inputs.")
        return y

    # Extract valid x values
    x_valid = x[valid_mask]
    y_valid = np.empty_like(x_valid)

    # Handle x > 1e7
    high_mask = x_valid > 1e7
    if np.any(high_mask):
        y_valid[high_mask] = 1 / np.sqrt(x_valid[high_mask])

    # Handle x < 1e-6
    low_mask = x_valid < 1e-6
    if np.any(low_mask):
        y_valid[low_mask] = 1 / x_valid[low_mask]

    # Remaining values for Newton's method
    newton_mask = ~(high_mask | low_mask)
    if np.any(newton_mask):
        x_newton = x_valid[newton_mask]
        y_newton = 0.5 + 1 / x_newton

        max_iter = 50
        tol = 1e-8

        for _ in range(max_iter):
            tri = special.polygamma(1, y_newton)
            psi_2 = special.polygamma(2, y_newton)
            dif = tri * (1 - tri / x_newton) / psi_2
            y_newton = y_newton + dif

            if np.max(np.abs(dif) / y_newton) < tol:
                break
        else:
            print("Warning: Iteration limit exceeded.")

        y_valid[newton_mask] = y_newton

    # Update the valid entries in y
    y[valid_mask] = y_valid

    if np.any(~valid_mask):
        print("Warning: NaNs produced due to invalid inputs.")

    return y

def trigammaInverse_legacy(x):
    """
    Compute the inverse of the trigamma function.

    This function uses an iterative method to find y such that trigamma(y) = x.

    Parameters:
    -----------
    x : array_like
        Input values for which to compute the inverse trigamma.

    Returns:
    --------
    y : ndarray
        The inverse trigamma of the input values.

    Notes:
    ------
    - The function uses Newton's method for optimization.
    - It handles invalid inputs (NaN or negative values) by setting the output to NaN.
    - The iteration stops when the change is less than a tolerance or after a maximum number of iterations.

    Warnings:
    ---------
    - Prints a warning if NaNs are produced due to invalid inputs.
    - Prints a warning if the maximum number of iterations is reached without convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([])

    # Initialize y with a copy of x
    y = np.copy(x)

    # Handle invalid values: NaNs and x < 0
    invalid = np.isnan(x) | (x < 0)
    if np.any(invalid):
        y[invalid] = np.nan
        print("Warning: NaNs produced")

    # Indices of valid entries
    valid_indices = np.where(~invalid)[0]
    x_valid = x[~invalid]
    y_valid = y[~invalid]

    if x_valid.size == 0:
        return y

    # Handle x > 1e7
    omit = x_valid > 1e7
    if np.any(omit):
        y_valid[omit] = 1 / np.sqrt(x_valid[omit])
        # Update the corresponding entries in y
        y[valid_indices[omit]] = y_valid[omit]
        # Remove these entries from further processing
        keep = ~omit
        x_valid = x_valid[keep]
        y_valid = y_valid[keep]
        valid_indices = valid_indices[keep]
        if x_valid.size == 0:
            return y

    # Handle x < 1e-6
    omit = x_valid < 1e-6
    if np.any(omit):
        y_valid[omit] = 1 / x_valid[omit]
        # Update the corresponding entries in y
        y[valid_indices[omit]] = y_valid[omit]
        # Remove these entries from further processing
        keep = ~omit
        x_valid = x_valid[keep]
        y_valid = y_valid[keep]
        valid_indices = valid_indices[keep]
        if x_valid.size == 0:
            return y

    # Initialize y_valid with a good starting value
    y_valid[:] = 0.5 + 1 / x_valid

    max_iter = 50
    tol = 1e-8
    for _ in range(max_iter):
        # Compute trigamma and tetragamma at current y_valid
        tri = special.polygamma(1, y_valid)
        psi_2 = special.polygamma(2, y_valid)

        # Newton's method update (matching the R function)
        dif = tri * (1 - tri / x_valid) / psi_2
        y_valid = y_valid + dif

        # Check for convergence
        if np.max(-dif / y_valid) < tol:
            break
    else:
        print("Warning: Iteration limit exceeded")

    # Update the corresponding entries in y
    y[valid_indices] = y_valid
    return y


def ebayes(fit, proportion=0.01, stdev_coef_lim=(0.1, 4)):
    """
    Perform empirical Bayes moderation of standard errors for linear model fits.

    This function implements the empirical Bayes method to moderate the standard errors
    of the coefficient estimates from linear model fits. It calculates moderated t-statistics,
    p-values, and log-odds of differential expression.

    Parameters:
    -----------
    fit : dict
        A dictionary containing the results of a linear model fit. It should include:
        - 'coefficients': array-like, estimated coefficients
        - 'stdev_unscaled': array-like, unscaled standard deviations of coefficients
        - 'sigma': array-like, residual standard deviations
        - 'df_residual': array-like, residual degrees of freedom
    proportion : float, optional
        The expected proportion of differentially expressed genes (default is 0.01)
    stdev_coef_lim : tuple of float, optional
        Limits for the prior standard deviation for the coefficients (default is (0.1, 4))

    Returns:
    --------
    dict
        A dictionary containing the results of the empirical Bayes analysis, including:
        - 's2_prior': prior variance
        - 's2_post': posterior variance
        - 'df_prior': prior degrees of freedom
        - 't': moderated t-statistics
        - 'p_value': p-values
        - 'var_prior': prior variance for coefficients
        - 'lods': log-odds of differential expression

    Raises:
    -------
    ValueError
        If the input fit dictionary is invalid or contains insufficient data

    Notes:
    ------
    This function implements the empirical Bayes method described in Smyth (2004).
    """
    # Extract components from the fit dictionary
    coefficients = fit.params
    stdev_unscaled = fit.normalized_cov_params
    sigma = fit.mse_resid
    df_residual = fit.df_resid

    # Check for valid input
    if coefficients is None or stdev_unscaled is None or sigma is None or df_residual is None:
        raise ValueError("No data, or argument is not a valid lmFit object")
    if np.all(df_residual == 0):
        raise ValueError("No residual degrees of freedom in linear model fits")
    if np.all(~np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")

    # Perform variance squeezing
    out = squeezeVar(sigma ** 2, df_residual)
    out.s2_prior = out.var_prior
    out.s2_post = out.var_post
    del out.var_prior, out.var_post

    # Calculate degrees of freedom and moderated t-statistic
    df_total = df_residual + out.df_prior
    s2_post_sqrt = np.sqrt(out.s2_post)[:, np.newaxis]
    t = coefficients / stdev_unscaled / s2_post_sqrt
    out.t = t

    # Calculate p-values
    out.p_value = 2 * stats.t.sf(np.abs(t), df=df_total[:, np.newaxis])

    # Calculate B-statistic (log-odds of differential expression)
    var_prior_lim = np.array(stdev_coef_lim) ** 2 / out.s2_prior
    out.var_prior = tmixture_matrix(t, stdev_unscaled, df_total, proportion, var_prior_lim)

    # Handle cases where var_prior estimation fails
    if np.any(np.isnan(out.var_prior)):
        out.var_prior[np.isnan(out.var_prior)] = 1 / out.s2_prior
        print("Warning: Estimation of var.prior failed - set to default value")

    # Calculate ratio of total to residual variance
    r = (stdev_unscaled ** 2 + out.var_prior) / stdev_unscaled ** 2
    t2 = t ** 2

    # Calculate log-odds of differential expression
    if out.df_prior > 1e6:
        kernel = t2 * (1 - 1 / r) / 2
    else:
        df_total_expanded = df_total[:, np.newaxis]
        kernel = ((1 + df_total_expanded) / 2) * np.log((t2 + df_total_expanded) / (t2 / r + df_total_expanded))
    lods = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel
    out.lods = lods

    return out


def tmixture_matrix(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    """
    Compute the prior variances for each coefficient using a t-mixture model.

    This function applies the t-mixture model to each column of the input matrices,
    estimating the prior variance for each coefficient.

    Parameters:
    -----------
    tstat : array-like
        Matrix of t-statistics, where each column corresponds to a coefficient.
    stdev_unscaled : array-like
        Matrix of unscaled standard deviations, with the same shape as tstat.
    df : array-like
        Degrees of freedom for each observation.
    proportion : float
        The proportion of genes expected to be differentially expressed.
    v0_lim : tuple of float, optional
        Limits for the prior variance. If provided, must be a tuple of length 2.

    Returns:
    --------
    v0 : ndarray
        Array of estimated prior variances for each coefficient.

    Raises:
    -------
    ValueError
        If the shapes of tstat and stdev_unscaled don't match, or if v0_lim is provided but not of length 2.
    """
    # Convert inputs to numpy arrays
    tstat = np.asarray(tstat)
    stdev_unscaled = np.asarray(stdev_unscaled)

    # Check input dimensions
    if tstat.shape != stdev_unscaled.shape:
        raise ValueError("Dims of tstat and stdev_unscaled don't match")
    if v0_lim is not None and len(v0_lim) != 2:
        raise ValueError("v0.lim must have length 2")

    # Get the number of coefficients (columns in tstat)
    ncoef = tstat.shape[1]

    # Initialize array to store prior variances
    v0 = np.zeros(ncoef)

    # Compute prior variance for each coefficient
    for j in range(ncoef):
        v0[j] = tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)

    return v0


def tmixture_vector(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    """
    Estimate the prior variance for a single coefficient using a t-mixture model.

    This function implements the core algorithm for estimating the prior variance
    in the empirical Bayes t-mixture model. It processes a single vector of t-statistics
    and corresponding standard deviations.

    Parameters:
    -----------
    tstat : array-like
        Vector of t-statistics for a single coefficient.
    stdev_unscaled : array-like
        Vector of unscaled standard deviations, same length as tstat.
    df : array-like
        Degrees of freedom for each observation.
    proportion : float
        The proportion of genes expected to be differentially expressed.
    v0_lim : tuple of float, optional
        Limits for the prior variance. If provided, must be a tuple of length 2.

    Returns:
    --------
    float
        Estimated prior variance for the coefficient.

    Notes:
    ------
    The function implements a complex algorithm involving several steps:
    1. Data preparation and filtering
    2. Calculation of target statistics
    3. Estimation of probabilities and quantiles
    4. Computation of prior variances
    5. Optional clipping of results
    """
    # Convert inputs to numpy arrays and remove NaN values
    tstat = np.asarray(tstat)
    stdev_unscaled = np.asarray(stdev_unscaled)
    df = np.asarray(df)
    valid = ~np.isnan(tstat)
    tstat = tstat[valid]
    stdev_unscaled = stdev_unscaled[valid]
    df = df[valid]

    # Calculate the number of genes and target genes
    ngenes = len(tstat)
    ntarget = int(np.ceil(proportion / 2 * ngenes))
    if ntarget < 1:
        return np.nan

    # Adjust the proportion if necessary
    p = max(ntarget / ngenes, proportion)

    # Compute absolute t-statistics and determine the threshold
    tstat_abs = np.abs(tstat)
    ttarget = np.percentile(tstat_abs, 100 * (ngenes - ntarget) / (ngenes - 1))

    # Select top genes based on the threshold
    top = tstat_abs >= ttarget
    tstat_top = tstat_abs[top]
    v1 = stdev_unscaled[top] ** 2
    df_top = df[top]

    # Compute ranks and probabilities
    r = ntarget - stats.rankdata(tstat_top, method='ordinal') + 1
    p0 = stats.t.cdf(-tstat_top, df=df_top)
    ptarget = ((r - 0.5) / (2 * ngenes) - (1 - p) * p0) / p

    # Estimate prior variances
    pos = ptarget > p0
    v0 = np.zeros(len(tstat_top))
    if np.any(pos):
        qtarget = stats.t.ppf(ptarget[pos], df=df_top[pos])
        v0[pos] = v1[pos] * ((tstat_top[pos] / qtarget) ** 2 - 1)

    # Clip prior variances if limits are provided
    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])

    # Return the mean of estimated prior variances
    return np.mean(v0)

from sklearn.linear_model import LinearRegression

class SklearnLinearModel:
    def __init__(self,n_jobs=-1):
        self.model = LinearRegression(fit_intercept=False,n_jobs=n_jobs)
        self.coefficients = None
        self.cov_params = None
        self.stdev_unscaled = None
        self.sigma = None
        self.df_residual = None

    def fit(self, X, y):
        """
        Fit the linear model to the data X and y.

        Parameters:
        X (numpy.ndarray or pandas.DataFrame): The input feature matrix.
        y (numpy.ndarray or pandas.Series): The target vector.
        """
        # Fit the linear model
        self.model.fit(X, y)
        
        # Store the coefficients
        self.coefficients = self.model.coef_
        
        # Compute (X'X)^-1
        XTX = np.dot(X.T, X)
        try:
            XTX_inv = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            # If XTX is singular, use pseudo-inverse
            XTX_inv = np.linalg.pinv(XTX)
        self.cov_params = XTX_inv
        
        # Compute standard deviations (sqrt of diagonal of (X'X)^-1)
        self.stdev_unscaled = np.sqrt(np.diag(self.cov_params))
        
        # Compute residuals
        predictions = self.model.predict(X)
        residuals = y - predictions
        mse_resid = np.sum(residuals**2,axis=0) / (X.shape[0] - X.shape[1])
        self.sigma = np.sqrt(mse_resid)
        
        # Degrees of freedom
        self.df_residual = X.shape[0] - X.shape[1]

    def get_coefficients(self):
        """Return the regression coefficients."""
        return self.coefficients

    def get_cov_params(self):
        """Return the covariance parameters (X'X)^-1."""
        return self.cov_params

    def get_stdev_unscaled(self):
        """Return the unscaled standard deviations of the coefficients."""
        return self.stdev_unscaled

    def get_sigma(self):
        """Return the residual standard deviation."""
        return self.sigma

    def get_df_residual(self):
        """Return the degrees of freedom of the residuals."""
        return self.df_residual

class FComputer():
    def __init__(self,n_jobs=-1):
        
        self.n_genes = None
        self.n_samples = None
        self.n_predictors = None
        self.coefficients = None
        self.stdev_unscaled = None
        self.sigma = None
        self.df_residual = None

        self.n_jobs = n_jobs

    def fit_sklearn(self,Y,X,gene_names=None):

        self.n_genes, self.n_samples = Y.shape
        self.gene_names = gene_names
        self.n_predictors = X.shape[1]

        model = SklearnLinearModel(n_jobs=self.n_jobs)
        model.fit(X, Y.T)
        
        self.coefficients = model.get_coefficients()
        self.stdev_unscaled = model.get_stdev_unscaled().reshape(1,-1).repeat(self.n_genes,axis=0)
        self.sigma = model.get_sigma()
        self.df_residual = np.repeat(model.get_df_residual(),self.n_genes)

    def fit(self,Y,X,gene_names=None):

        self.n_genes, self.n_samples = Y.shape
        self.gene_names = gene_names
        self.n_predictors = X.shape[1]
        self.coefficients = np.zeros((self.n_genes, self.n_predictors))
        self.stdev_unscaled = np.zeros((self.n_genes, self.n_predictors))
        self.sigma = np.zeros(self.n_genes)
        self.df_residual = np.zeros(self.n_genes)

        # Fit linear models for each gene
        for i in tqdm(range(self.n_genes), desc="Fitting linear models"):
            y = Y[i, :]
            model = sm.OLS(y, X)
            results = model.fit()
            self.coefficients[i, :] = results.params
            cov_params = results.normalized_cov_params  # (X'X)^{-1}
            self.stdev_unscaled[i, :] = np.sqrt(np.diag(cov_params))
            self.sigma[i] = np.sqrt(results.mse_resid)       # Residual standard deviation
            self.df_residual[i] = results.df_resid

    def eBayes(self, proportion=0.01, stdev_coef_lim=(0.1, 4)):
        
        # Extract components from the fit dictionary
        coefficients = self.coefficients
        stdev_unscaled = self.stdev_unscaled
        sigma = self.sigma
        df_residual = self.df_residual

        # Check for valid input
        if coefficients is None or stdev_unscaled is None or sigma is None or df_residual is None:
            raise ValueError("No data, or argument is not a valid lmFit object")
        if np.all(df_residual == 0):
            raise ValueError("No residual degrees of freedom in linear model fits")
        if np.all(~np.isfinite(sigma)):
            raise ValueError("No finite residual standard deviations")

        # Perform variance squeezing
        out = squeezeVar(sigma ** 2, df_residual)
        self.s2_prior = out['var_prior']
        self.s2_post = out['var_post']
        self.df_prior = out['df_prior']
        del out['var_prior'], out['var_post'], out['df_prior']

        # Calculate degrees of freedom and moderated t-statistic
        df_total = df_residual + self.df_prior
        s2_post_sqrt = np.sqrt(self.s2_post)[:, np.newaxis]
        t = coefficients / stdev_unscaled / s2_post_sqrt
        self.t = t

        # Calculate p-values
        self.p_value = 2 * stats.t.sf(np.abs(t), df=df_total[:, np.newaxis])

        # Calculate B-statistic (log-odds of differential expression)
        var_prior_lim = np.array(stdev_coef_lim) ** 2 / self.s2_prior
        self.var_prior = tmixture_matrix(t, stdev_unscaled, df_total, proportion, var_prior_lim)

        # Handle cases where var_prior estimation fails
        if np.any(np.isnan(self.var_prior)):
            self.var_prior[np.isnan(self.var_prior)] = 1 / self.s2_prior
            print("Warning: Estimation of var.prior failed - set to default value")

        # Calculate ratio of total to residual variance
        r = (stdev_unscaled ** 2 + self.var_prior) / stdev_unscaled ** 2
        t2 = t ** 2

        # Calculate log-odds of differential expression
        if self.df_prior > 1e6:
            kernel = t2 * (1 - 1 / r) / 2
        else:
            df_total_expanded = df_total[:, np.newaxis]
            kernel = ((1 + df_total_expanded) / 2) * np.log((t2 + df_total_expanded) / (t2 / r + df_total_expanded))
        lods = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel
        self.lods = lods

    def Fstats(self):
        self.F = np.sum(self.t**2, axis=1)