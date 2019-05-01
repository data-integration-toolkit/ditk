\name{ptmvt}
\alias{ptmvt}
\title{Truncated Multivariate Student t Distribution}
\description{
Computes the distribution function of the truncated multivariate t 
distribution
}
\usage{
ptmvt(lowerx, upperx, mean = rep(0, length(lowerx)), sigma, df = 1, 
  lower = rep(-Inf, length = length(mean)), 
  upper = rep(Inf, length = length(mean)), maxpts = 25000, abseps = 0.001, 
  releps = 0)
}

\arguments{
  \item{lowerx}{ the vector of lower limits of length n.}
  \item{upperx}{ the vector of upper limits of length n.}
  \item{mean}{ the mean vector of length n.}
  \item{sigma}{ the covariance matrix of dimension n. Either \code{corr} or
                \code{sigma} can be specified. If \code{sigma} is given, the
                problem is standardized. If neither \code{corr} nor
                \code{sigma} is given, the identity matrix is used 
                for \code{sigma}. }
  \item{df}{Degrees of freedom parameter}              
  \item{lower}{Vector of lower truncation points, 
                 default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points, 
                 default is \code{rep( Inf, length = length(mean))}.}              
  \item{maxpts}{ maximum number of function values as integer. }
  \item{abseps}{ absolute error tolerance as double. }
  \item{releps}{ relative error tolerance as double. }
}
\value{
The evaluated distribution function is returned with attributes
  \item{error}{estimated absolute error and}
  \item{msg}{status messages.}
}
\references{
Geweke, J. F. (1991) Efficient simulation from the multivariate normal and Student-t distributions 
 subject to linear constraints and the evaluation of constraint probabilities.
 \url{http://www.biz.uiowa.edu/faculty/jgeweke/papers/paper47/paper47.pdf} 
 
Samuel Kotz, Saralees Nadarajah (2004). Multivariate t Distributions and Their Applications. 
  \emph{Cambridge University Press}
}
\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>}

\examples{
sigma <- matrix(c(5, 0.8, 0.8, 1), 2, 2)
Fx <- ptmvt(lowerx=c(-1,-1), upperx=c(0.5,0), mean=c(0,0), sigma=sigma, df=3, 
  lower=c(-1,-1), upper=c(1,1))
}

\keyword{ math }
\keyword{ multivariate }
