\name{ptmvtnorm.marginal}
\Rdversion{1.1}
\alias{ptmvnorm.marginal}
\alias{ptmvt.marginal}
\title{One-dimensional marginal CDF function for a Truncated Multivariate Normal and Student t distribution}
\description{
  This function computes the one-dimensional marginal probability 
  function from a Truncated Multivariate Normal and Student t density function
  using integration in \code{pmvnorm()} and \code{pmvt()}.
}
\usage{
ptmvnorm.marginal(xn, 
    n = 1, 
    mean = rep(0, nrow(sigma)), 
    sigma = diag(length(mean)), 
    lower = rep(-Inf, length = length(mean)), 
    upper = rep(Inf, length = length(mean)))
ptmvt.marginal(xn, 
    n = 1, 
    mean = rep(0, nrow(sigma)), 
    sigma = diag(length(mean)), 
    df = 1, 
    lower = rep(-Inf, length = length(mean)), 
    upper = rep(Inf, length = length(mean)))    
}
\arguments{
  \item{xn}{Vector of quantiles to calculate the marginal probability for.}
  \item{n}{Index position (1..k) within the random vector xn to calculate the one-dimensional marginal probability for.}
  \item{mean}{ the mean vector of length k. }
  \item{sigma}{ the covariance matrix of dimension k. Either \code{corr} or
                \code{sigma} can be specified. If \code{sigma} is given, the
                problem is standardized. If neither \code{corr} nor
                \code{sigma} is given, the identity matrix is used
                for \code{sigma}.}
  \item{df}{degrees of freedom parameter}              
  \item{lower}{Vector of lower truncation points, 
                 default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points, 
                 default is \code{rep( Inf, length = length(mean))}.}
}

\details{
  The one-dimensional marginal probability for index i is 
  \eqn{F_i(x_i) = P(X_i \le x_i)}{F_i(x_i) = P(X_i <= x_i)}
  \deqn{F_i(x_i) = \int_{a_1}^{b_1} \ldots \int_{a_{i-1}}^{b_{i-1}} \int_{a_{i}}^{x_i} \int_{a_{i+1}}^{b_{i+1}} \ldots \int_{a_k}^{b_k} f(x) dx 
                 = \alpha^{-1} \Phi_k(a, u, \mu, \Sigma)}
  where \eqn{u = (b_1,\ldots,b_{i-1},x_i,b_{i+1},\ldots,b_k)'}{u = (b_1,...,b_{i-1},x_i,b_{i+1},...,b_k)'} 
  is the upper integration bound and \eqn{\Phi_k} 
  is the k-dimensional normal probability (i.e. functions \code{pmvnorm()} and \code{pmvt()} in R package \code{mvtnorm}).  
}

\value{
 Returns a vector of the same length as xn with probabilities.
}
\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>}

\examples{
## Example 1: Truncated multi-normal
lower <- c(-1,-1,-1)
upper <- c(1,1,1)
mean <- c(0,0,0)
sigma <- matrix(c(  1, 0.8, 0.2, 
                  0.8,   1, 0.1,
                  0.2, 0.1,   1), 3, 3)

X <- rtmvnorm(n=1000, mean=c(0,0,0), sigma=sigma, lower=lower, upper=upper)

x <- seq(-1, 1, by=0.01)
Fx <- ptmvnorm.marginal(xn=x, n=1, mean=c(0,0,0), sigma=sigma, lower=lower, upper=upper) 

plot(ecdf(X[,1]), main="marginal CDF for truncated multi-normal")
lines(x, Fx, type="l", col="blue")

## Example 2: Truncated multi-t
X <- rtmvt(n=1000, mean=c(0,0,0), sigma=sigma, df=2, lower=lower, upper=upper)

x <- seq(-1, 1, by=0.01)
Fx <- ptmvt.marginal(xn=x, n=1, mean=c(0,0,0), sigma=sigma, lower=lower, upper=upper) 

plot(ecdf(X[,1]), main="marginal CDF for truncated multi-t")
lines(x, Fx, type="l", col="blue")
}

\keyword{distribution}
\keyword{multivariate}
