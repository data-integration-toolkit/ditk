\name{qmvnorm}
\alias{qmvnorm}
\title{ Quantiles of the Multivariate Normal Distribution }
\description{

Computes the equicoordinate quantile function of the multivariate normal
distribution for arbitrary correlation matrices
based on inversion of \code{\link{pmvnorm}}.

}
\usage{
qmvnorm(p, interval = NULL, tail = c("lower.tail", 
        "upper.tail", "both.tails"), mean = 0, corr = NULL, 
        sigma = NULL, algorithm = GenzBretz(),
        ptol = 0.001, maxiter = 500, trace = FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{p}{ probability.}
  \item{interval}{ optional, a vector containing the end-points of the
    interval to be searched. Does not need to contain the true quantile,
    just used as starting values by the root-finder. If equal to NULL
    a guess is used.}
  \item{tail}{ specifies which quantiles should be computed.
               \code{lower.tail} gives the quantile \eqn{x} for which 
               \eqn{P[X \le x] = p}, \code{upper.tail} gives \eqn{x} with 
               \eqn{P[X > x] = p} and 
               \code{both.tails} leads to \eqn{x} 
               with \eqn{P[-x \le X \le x] = p}.}
  \item{mean}{ the mean vector of length n. }
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
  \item{ptol, maxiter, trace}{Parameters passed to the stochastic root-finding
    algorithm. Iteration stops when the 95\% confidence interval
    for the predicted quantile is inside [p-ptol, p+ptol]. \code{maxiter} is the
    maximum number of iterations for the root finding algorithm. \code{trace}
    prints the iterations of the root finder.}
  \item{...}{ additional parameters to be passed to 
              \code{\link{GenzBretz}}.}
}
\details{

  Only equicoordinate quantiles are computed, i.e., the quantiles in each
  dimension coincide. The result is seed dependend.
}
\value{
  A list with two components: \code{quantile} and \code{f.quantile} 
  give the location of the quantile and the difference between the distribution
  function evaluated at the quantile and \code{p}. 
}
\seealso{\code{\link{pmvnorm}}, \code{\link{qmvt}}}
\examples{
qmvnorm(0.95, sigma = diag(2), tail = "both")
}
\keyword{distribution}
