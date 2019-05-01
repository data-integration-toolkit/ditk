\name{mle.tmvnorm}
\alias{mle.tmvnorm}
\title{
Maximum Likelihood Estimation for the Truncated Multivariate Normal Distribution
}
\description{
Maximum Likelihood Estimation for the Truncated Multivariate Normal Distribution
}
\usage{
mle.tmvnorm(X, 
  lower = rep(-Inf, length = ncol(X)), 
  upper = rep(+Inf, length = ncol(X)), 
  start = list(mu = rep(0, ncol(X)), sigma = diag(ncol(X))), 
  fixed = list(), method = "BFGS", 
  cholesky = FALSE,
  lower.bounds = -Inf,
  upper.bounds = +Inf,
  ...)
}
\arguments{
  \item{X}{Matrix of quantiles, each row is taken to be a quantile.}
  \item{lower}{Vector of lower truncation points, 
              default is \code{rep(-Inf, length = ncol(X))}.}
  \item{upper}{Vector of upper truncation points, 
              default is \code{rep( Inf, length = ncol(X))}.}
  \item{start}{Named list with elements \code{mu} (mean vector) and \code{sigma} (covariance matrix). Initial values for optimizer.}
  \item{fixed}{Named list. Parameter values to keep fixed during optimization.}
  \item{method}{Optimization method to use. See \code{\link{optim}}}
  \item{cholesky}{if TRUE, we use the Cholesky decomposition of \code{sigma} as parametrization}
  \item{lower.bounds}{lower bounds/box constraints for method "L-BFGS-B"}
  \item{upper.bounds}{upper bounds/box constraints for method "L-BFGS-B"}
  \item{\dots}{Further arguments to pass to \code{\link{optim}}}
}
\details{
 This method performs a maximum likelihood estimation of the parameters \code{mean} and \code{sigma} of a truncated multinormal distribution, 
 when the truncation points \code{lower} and \code{upper} are known.
 \code{mle.tmvnorm()} is a wrapper for the general maximum likelihood method \code{\link[stats4]{mle}},
 so one does not have to specify the negative log-likelihood function.
 
 The log-likelihood function for a data matrix X (T x n) can be established straightforward as
\deqn{
\log L(X | \mu,\Sigma) = -T \log{\alpha(\mu,\Sigma)} + {-T/2} \log{\|\Sigma\|} -\frac{1}{2} \sum_{t=1}^{T}{(x_t-\mu)' \Sigma^{-1} (x_t-\mu)}
}

 As \code{\link[stats4]{mle}}, this method returns an object of class \code{mle}, for which various
 diagnostic methods are available, like \code{profile()}, \code{confint()} etc. See examples.
 
 In order to adapt the estimation problem to \code{\link[stats4]{mle}}, the named parameters 
 for mean vector elements are "mu_i" and the elements of the covariance matrix are "sigma_ij" for the lower triangular matrix elements, 
 i.e. (j <= i).
}

\value{
An object of class \code{\link[stats4]{mle-class}}
}

\author{
Stefan Wilhelm \email{wilhelm@financial.com}
}

\seealso{
\code{\link[stats4]{mle}} and \code{\link[stats4]{mle-class}}
}

\examples{
\dontrun{
set.seed(1.2345)

# the actual parameters
lower <- c(-1,-1)
upper <- c(1, 2)
mu    <- c(0, 0)
sigma <- matrix(c(1, 0.7,
               0.7, 2), 2, 2)
               
# generate random samples               
X <- rtmvnorm(n=500, mu, sigma, lower, upper)
method <- "BFGS"

# estimate mean vector and covariance matrix sigma from random samples X
# with default start values
mle.fit1 <- mle.tmvnorm(X, lower=lower, upper=upper)

# diagnostic output of the estimated parameters
summary(mle.fit1)
logLik(mle.fit1)
vcov(mle.fit1)

# profiling the log likelihood and confidence intervals
mle.profile1 <- profile(mle.fit1, X, method="BFGS", trace=TRUE)
confint(mle.profile1)

par(mfrow=c(3,2))
plot(mle.profile1)

# choosing a different start value
mle.fit2 <- mle.tmvnorm(X, lower=lower, upper=upper, 
  start=list(mu=c(0.1, 0.1), 
  sigma=matrix(c(1, 0.4, 0.4, 1.8),2,2)))
summary(mle.fit2)
}
}