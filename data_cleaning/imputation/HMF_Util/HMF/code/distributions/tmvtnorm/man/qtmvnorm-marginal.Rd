\encoding{latin1}
\name{qtmvnorm-marginal}
\alias{qtmvnorm.marginal}
\title{ Quantiles of the Truncated Multivariate Normal Distribution in one dimension}
\description{

Computes the equicoordinate quantile function of the truncated multivariate normal
distribution for arbitrary correlation matrices
based on an inversion of the algorithms by Genz and Bretz.
}
\usage{
qtmvnorm.marginal(p, 
        interval = c(-10, 10), 
        tail = c("lower.tail","upper.tail","both.tails"), 
        n=1, 
		mean=rep(0, nrow(sigma)), 
		sigma=diag(length(mean)), 
		lower=rep(-Inf, length = length(mean)), 
		upper=rep( Inf, length = length(mean)),
		...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{p}{ probability.}
  \item{interval}{ a vector containing the end-points of the interval to be
          searched by \code{\link{uniroot}}.}
  \item{tail}{ specifies which quantiles should be computed.
               \code{lower.tail} gives the quantile \eqn{x} for which 
               \eqn{P[X \le x] = p}{P[X <= x] = p}, \code{upper.tail} gives \eqn{x} with 
               \eqn{P[X > x] = p} and 
               \code{both.tails} leads to \eqn{x} 
               with \eqn{P[-x \le X \le x] = p}.}{P[-x <= X <= x] = p}
  \item{n}{ index (1..n) to calculate marginal quantile for}            
  \item{mean}{ the mean vector of length n. }
  \item{sigma}{ the covariance matrix of dimension n. Either \code{corr} or
                \code{sigma} can be specified. If \code{sigma} is given, the
                problem is standardized. If neither \code{corr} nor
                \code{sigma} is given, the identity matrix is used
                for \code{sigma}. }
  \item{lower}{Vector of lower truncation points,\\ 
                 default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points,\\ 
                 default is \code{rep( Inf, length = length(mean))}.}              
  \item{...}{ additional parameters to be passed to
              \code{\link{uniroot}}.}
}
\details{

  Only equicoordinate quantiles are computed, i.e., the quantiles in each
  dimension coincide. Currently, the distribution function is inverted by 
  using the \code{\link{uniroot}} function which may result in limited accuracy of the 
  quantiles. 
}
\value{
  A list with four components: \code{quantile} and \code{f.quantile} 
  give the location of the quantile and the value of the function
  evaluated at that point. \code{iter} and \code{estim.prec} give the number 
  of iterations used and an approximate estimated precision from
  \code{\link{uniroot}}.
}
\seealso{\code{\link{ptmvnorm}}, \code{\link[mvtnorm]{pmvnorm}}}
\examples{
# finite dimensional distribution of the Geometric Brownian Motion log-returns 
# with truncation

# volatility p.a.
sigma=0.4

# risk free rate
r = 0.05

# n=3 points in time
T <- c(0.5, 0.7, 1)

# covariance matrix of Geometric Brownian Motion returns
Sigma = sigma^2*outer(T,T,pmin)

# mean vector of the Geometric Brownian Motion returns
mu    = (r - sigma^2/2) * T

# lower truncation vector a (a<=x<=b)
a = rep(-Inf, 3)

# upper truncation vector b (a<=x<=b)
b = c(0, 0, Inf)

# quantile of the t_1 returns
qtmvnorm.marginal(p=0.95, interval = c(-10, 10), tail = "lower.tail", n=1, 
  mean  = mu, sigma = Sigma, lower=a, upper=b)
}

\keyword{distribution}
\keyword{multivariate}