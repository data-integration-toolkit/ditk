\name{mtmvnorm}
\alias{mtmvnorm}
\alias{moments}

\title{Computation of Mean Vector and Covariance Matrix For Truncated Multivariate Normal Distribution}
\description{
  Computation of the first two moments, i.e. mean vector and covariance matrix 
  for the Truncated Multivariate Normal Distribution 
  based on the works of Tallis (1961), Lee (1979) and Leppard and Tallis (1989),
  but extended to the double-truncated case with general mean and general covariance matrix.
}
\usage{
mtmvnorm(mean = rep(0, nrow(sigma)), 
  sigma = diag(length(mean)), 
  lower = rep(-Inf, length = length(mean)), 
  upper = rep(Inf, length = length(mean)), 
  doComputeVariance=TRUE,
  pmvnorm.algorithm=GenzBretz())
}

\arguments{
  \item{mean}{Mean vector, default is \code{rep(0, length = ncol(x))}.}
  \item{sigma}{Covariance matrix, default is \code{diag(ncol(x))}.}
  \item{lower}{Vector of lower truncation points,\\ 
              default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points,\\ 
              default is \code{rep( Inf, length = length(mean))}.}
  \item{doComputeVariance}{flag whether to compute the variance for users who are interested only in the mean. 
                           Defaults to \code{TRUE} for backward compatibility.}
  \item{pmvnorm.algorithm}{Algorithm used for \code{\link[mvtnorm]{pmvnorm}}}
}

\details{
Details for the moment calculation under double truncation and the derivation of the formula
can be found in the Manjunath/Wilhelm (2009) working paper.
If only a subset of variables are truncated, we calculate the truncated moments only for these
and use the Johnson/Kotz formula for the remaining untruncated variables.
}

\value{
  \item{tmean}{Mean vector of truncated variables}
  \item{tvar}{Covariance matrix of truncated variables}
}

\references{

Tallis, G. M. (1961). The moment generating function of the truncated multinormal distribution. 
\emph{Journal of the Royal Statistical Society, Series B}, \bold{23}, 223--229

Johnson, N./Kotz, S. (1970). Distributions in Statistics: Continuous Multivariate Distributions
\emph{Wiley & Sons}, pp. 70--73

Lee, L.-F. (1979). On the first and second moments of the truncated multi-normal distribution and a simple estimator. 
\emph{Economics Letters}, \bold{3}, 165--169

Leppard, P. and Tallis, G. M. (1989). Evaluation of the Mean and Covariance of the Truncated Multinormal.
\emph{Applied Statistics}, \bold{38}, 543--553

Manjunath B G and Wilhelm, S. (2009). Moments Calculation For the Double Truncated Multivariate Normal Density. 
Working Paper. Available at SSRN: \url{http://ssrn.com/abstract=1472153}
}

\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>, Manjunath B G <bgmanjunath@gmail.com>}

\examples{
  mu    <- c(0.5, 0.5, 0.5)
  sigma <- matrix(c(  1,  0.6, 0.3,
                    0.6,    1, 0.2,
                    0.3,  0.2,   2), 3, 3)
                    
  a  <- c(-Inf, -Inf, -Inf)
  b  <- c(1, 1, 1)

  # compute first and second moments
  mtmvnorm(mu, sigma, lower=a, upper=b)
  
  # compare with simulated results
  X <- rtmvnorm(n=1000, mean=mu, sigma=sigma, lower=a, upper=b)
  colMeans(X)
  cov(X)
}

\keyword{distribution}
\keyword{multivariate}
