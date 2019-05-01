\name{pmvnorm}
\alias{pmvnorm}
\title{ Multivariate Normal Distribution }
\description{

Computes the distribution function of the multivariate normal 
distribution for arbitrary limits and correlation matrices.

}
\usage{
pmvnorm(lower=-Inf, upper=Inf, mean=rep(0, length(lower)),
        corr=NULL, sigma=NULL, algorithm = GenzBretz(), ...)
}
\arguments{
  \item{lower}{ the vector of lower limits of length n.}
  \item{upper}{ the vector of upper limits of length n.}
  \item{mean}{ the mean vector of length n.}
  \item{corr}{ the correlation matrix of dimension n.}
  \item{sigma}{ the covariance matrix of dimension n. Either \code{corr} or
                \code{sigma} can be specified. If \code{sigma} is given, the
                problem is standardized. If neither \code{corr} nor
                \code{sigma} is given, the identity matrix is used 
                for \code{sigma}. }
  \item{algorithm}{ an object of class \code{\link{GenzBretz}},
                    \code{\link{Miwa}} or \code{\link{TVPACK}}
                    specifying both the algorithm to be used as well as 
                    the associated hyper parameters.}
  \item{...}{ additional parameters (currently given to \code{GenzBretz} for 
              backward compatibility issues). }
}
\details{

This program involves the computation of 
multivariate normal probabilities with arbitrary correlation matrices.
It involves both the computation of singular and nonsingular 
probabilities. The implemented methodology is described in
Genz (1992, 1993) (for algorithm GenzBretz), in Miwa et al. (2003)
for algorithm Miwa (useful up to dimension 20) and Genz (2004)
for the TVPACK algorithm (which covers 2- and 3-dimensional problems
for semi-infinite integration regions).

Note the default algorithm GenzBretz is randomized and hence slightly depends on 
\code{\link{.Random.seed}} and that both \code{-Inf} and \code{+Inf} may
be specified in \code{lower} and \code{upper}.  For more details see
\code{\link{pmvt}}.  

The multivariate normal 
case is treated as a special case of \code{\link{pmvt}} with \code{df=0} and 
univariate problems are passed to \code{\link{pnorm}}.

The multivariate normal density and random deviates are available using
\code{\link{dmvnorm}} and \code{\link{rmvnorm}}.
 
}
\value{
The evaluated distribution function is returned with attributes
  \item{error}{estimated absolute error and}
  \item{msg}{status messages.}
}
\references{ 

Genz, A. (1992). Numerical computation of multivariate normal probabilities.
\emph{Journal of Computational and Graphical Statistics}, \bold{1}, 141--150.

Genz, A. (1993). Comparison of methods for the computation of multivariate
normal probabilities. \emph{Computing Science and Statistics}, \bold{25},
400--405.

Genz, A. (2004), Numerical computation of rectangular bivariate and
trivariate normal and t-probabilities, \emph{Statistics and
Computing}, \bold{14}, 251--260.

Genz, A. and Bretz, F. (2009), \emph{Computation of Multivariate Normal and
t Probabilities}. Lecture Notes in Statistics, Vol. 195. Springer-Verlag,  
Heidelberg.

Miwa, A., Hayter J. and Kuriki, S. (2003).
The evaluation of general non-centred orthant probabilities.
\emph{Journal of the Royal Statistical Society}, Ser. B, 65, 223--234.

}

\source{
  \url{http://www.sci.wsu.edu/math/faculty/genz/homepage}
}

\seealso{\code{\link{qmvnorm}}}

\examples{

n <- 5
mean <- rep(0, 5)
lower <- rep(-1, 5)
upper <- rep(3, 5)
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
corr[upper.tri(corr)] <- 0.5
prob <- pmvnorm(lower, upper, mean, corr)
print(prob)

stopifnot(pmvnorm(lower=-Inf, upper=3, mean=0, sigma=1) == pnorm(3))

a <- pmvnorm(lower=-Inf,upper=c(.3,.5),mean=c(2,4),diag(2))

stopifnot(round(a,16) == round(prod(pnorm(c(.3,.5),c(2,4))),16))

a <- pmvnorm(lower=-Inf,upper=c(.3,.5,1),mean=c(2,4,1),diag(3))

stopifnot(round(a,16) == round(prod(pnorm(c(.3,.5,1),c(2,4,1))),16))

# Example from R News paper (original by Genz, 1992):

m <- 3
sigma <- diag(3)
sigma[2,1] <- 3/5
sigma[3,1] <- 1/3
sigma[3,2] <- 11/15
pmvnorm(lower=rep(-Inf, m), upper=c(1,4,2), mean=rep(0, m), corr=sigma)

# Correlation and Covariance

a <- pmvnorm(lower=-Inf, upper=c(2,2), sigma = diag(2)*2)
b <- pmvnorm(lower=-Inf, upper=c(2,2)/sqrt(2), corr=diag(2))
stopifnot(all.equal(round(a,5) , round(b, 5)))

}
\keyword{distribution}
