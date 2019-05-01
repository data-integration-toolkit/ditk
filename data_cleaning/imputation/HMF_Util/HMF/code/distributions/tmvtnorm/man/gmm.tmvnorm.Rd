\name{gmm.tmvnorm}
\alias{gmm.tmvnorm}
\title{
GMM Estimation for the Truncated Multivariate Normal Distribution
}
\description{
Generalized Method of Moments (GMM) Estimation for the Truncated Multivariate Normal Distribution
}
\usage{
gmm.tmvnorm(X, 
  lower = rep(-Inf, length = ncol(X)), 
  upper = rep(+Inf, length = ncol(X)), 
  start = list(mu = rep(0, ncol(X)), sigma = diag(ncol(X))), 
  fixed = list(),
  method=c("ManjunathWilhelm","Lee"),
  cholesky = FALSE,
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
  \item{method}{Which set of moment conditions used, possible methods are "ManjunathWilhelm" (default) and "Lee".}
  \item{cholesky}{if TRUE, we use the Cholesky decomposition of \code{sigma} as parametrization}
  \item{\dots}{Further arguments to pass to \code{\link{gmm}}}
}

\details{
 This method performs an estimation of the parameters \code{mean} and \code{sigma} of a truncated multinormal distribution
 using the Generalized Method of Moments (GMM), 
 when the truncation points \code{lower} and \code{upper} are known.
 \code{gmm.tmvnorm()} is a wrapper for the general GMM method \code{\link[gmm]{gmm}},
 so one does not have to specify the moment conditions.
 
 \bold{Manjunath/Wilhelm moment conditions}\cr
 Because the first and second moments can be computed thanks to the \code{\link{mtmvnorm}}
 function, we can set up a method-of-moments estimator by equating the sample moments
 to their population counterparts. This way we have an exactly identified case.
 
 \bold{Lee (1979,1983) moment conditions}\cr
 The recursive moment conditions presented by Lee (1979,1983) are defined for \eqn{l=0,1,2,\ldots}
 as
 \deqn{
  \sigma^{iT} E(x_i^l \textbf{x}) = \sigma^{iT} \mu E(x_i^l) + l E(x_i^{l-1}) + \frac{a_i^l F_i(a_i)}{F} - \frac{b_i^l F_i(b_i)}{F}
 }
 where \eqn{E(x_i^l)} and \eqn{E(x_i^l \textbf{x})} are the moments of \eqn{x_i^l} and \eqn{x_i^l \textbf{x}} 
 respectively and \eqn{F_i(c)/F} is the one-dimensional marginal density in variable \eqn{i} as calculated by \code{\link{dtmvnorm.marginal}}. 
 \eqn{\sigma^{iT}} is the \eqn{i}-th column of the inverse covariance matrix \eqn{\Sigma^{-1}}.
 
 This method returns an object of class \code{gmm}, for which various
 diagnostic methods are available, like \code{profile()}, \code{confint()} etc. See examples.
}

\value{
An object of class \code{\link[gmm]{gmm}}
}

\author{
Stefan Wilhelm \email{wilhelm@financial.com}
}

\references{
Tallis, G. M. (1961). The moment generating function of the truncated multinormal distribution. 
\emph{Journal of the Royal Statistical Society, Series B}, \bold{23}, 223--229

Lee, L.-F. (1979). On the first and second moments of the truncated multi-normal distribution and a simple estimator. 
\emph{Economics Letters}, \bold{3}, 165--169

Lee, L.-F. (1983). The determination of moments of the doubly truncated multivariate normal Tobit model.
\emph{Economics Letters}, \bold{11}, 245--250

Manjunath B G and Wilhelm, S. (2009). Moments Calculation For the Double Truncated Multivariate Normal Density. 
Working Paper. Available at SSRN: \url{http://ssrn.com/abstract=1472153}
 
}

\seealso{
\code{\link[gmm]{gmm}}
}

\examples{
\dontrun{
set.seed(1.234)

# the actual parameters
lower <- c(-1, -2)
upper <- c(3, Inf)
mu    <- c(0, 0)
sigma <- matrix(c(1, 0.8,
                0.8, 2), 2, 2)
               
# generate random samples               
X <- rtmvnorm(n=500, mu, sigma, lower, upper)

# estimate mean vector and covariance matrix sigma from random samples X
# with default start values
gmm.fit1 <- gmm.tmvnorm(X, lower=lower, upper=upper)

# diagnostic output of the estimated parameters
summary(gmm.fit1)
vcov(gmm.fit1)

# confidence intervals
confint(gmm.fit1)

# choosing a different start value
gmm.fit2 <- gmm.tmvnorm(X, lower=lower, upper=upper, 
  start=list(mu=c(0.1, 0.1), 
  sigma=matrix(c(1, 0.4, 0.4, 1.8),2,2)))
summary(gmm.fit2)

# GMM estimation with Lee (1983) moment conditions
gmm.fit3 <- gmm.tmvnorm(X, lower=lower, upper=upper, method="Lee")
summary(gmm.fit3)
confint(gmm.fit3)

# MLE estimation for comparison
mle.fit1 <- mle.tmvnorm(X, lower=lower, upper=upper)
confint(mle.fit1)
}
}
