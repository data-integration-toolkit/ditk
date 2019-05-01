% --- Source file: ptmvnorm.Rd ---
\name{ptmvnorm}
\alias{ptmvnorm}
\title{ Truncated Multivariate Normal Distribution }
\description{

Computes the distribution function of the truncated multivariate normal 
distribution for arbitrary limits and correlation matrices 
based on the \code{pmvnorm()} implementation of the algorithms by Genz and Bretz.

}
\usage{
ptmvnorm(lowerx, upperx, mean=rep(0, length(lowerx)), sigma, 
  lower = rep(-Inf, length = length(mean)), 
  upper = rep( Inf, length = length(mean)), 
  maxpts = 25000, abseps = 0.001, releps = 0)
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
  \item{lower}{Vector of lower truncation points,\\ 
                 default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points,\\ 
                 default is \code{rep( Inf, length = length(mean))}.}              
  \item{maxpts}{ maximum number of function values as integer. }
  \item{abseps}{ absolute error tolerance as double. }
  \item{releps}{ relative error tolerance as double. }
}
\details{

The computation of truncated multivariate normal probabilities and densities is done using conditional probabilities 
from the standard/untruncated multivariate normal distribution. 
So we refer to the documentation of the \code{mvtnorm} package and the methodology is described in
Genz (1992, 1993) and Genz/Bretz (2009).

For properties of the truncated multivariate normal distribution see for example Johnson/Kotz (1970)
and Horrace (2005).
}
\value{
The evaluated distribution function is returned with attributes
  \item{error}{estimated absolute error and}
  \item{msg}{status messages.}
}
\references{ 

Genz, A. (1992). Numerical computation of multivariate normal probabilities.
\emph{Journal of Computational and Graphical Statistics}, \bold{1}, 141--150 

Genz, A. (1993). Comparison of methods for the computation of multivariate
normal probabilities. \emph{Computing Science and Statistics}, \bold{25},
400--405

Genz, A. and Bretz, F. (2009). Computation of Multivariate Normal and t Probabilities. 
\emph{Lecture Notes in Statistics}, Vol. \bold{195}, Springer-Verlag, Heidelberg.

Johnson, N./Kotz, S. (1970). Distributions in Statistics: Continuous Multivariate Distributions
\emph{Wiley & Sons}, pp. 70--73

Horrace, W. (2005). Some Results on the Multivariate Truncated Normal Distribution. 
\emph{Journal of Multivariate Analysis}, \bold{94}, 209--221
}

\examples{
 sigma <- matrix(c(5, 0.8, 0.8, 1), 2, 2)
 Fx <- ptmvnorm(lowerx=c(-1,-1), upperx=c(0.5,0), mean=c(0,0), 
   sigma=sigma, lower=c(-1,-1), upper=c(1,1))
}

\keyword{distribution}
\keyword{multivariate}