\name{Mvnorm}
\alias{dmvnorm}
\alias{rmvnorm}
\title{Multivariate Normal Density and Random Deviates}
\description{
  These functions provide the density function and a random number
  generator for the multivariate normal
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma}.
}
\usage{
dmvnorm(x, mean = rep(0, p), sigma = diag(p), log = FALSE)
rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = diag(length(mean)),
        method=c("eigen", "svd", "chol"), pre0.9_9994 = FALSE)
}
\arguments{
 \item{x}{vector or matrix of quantiles. If \code{x} is a matrix, each
   row is taken to be a quantile.}
 \item{n}{number of observations.}
 \item{mean}{mean vector, default is \code{rep(0, length = ncol(x))}.}
 \item{sigma}{covariance matrix, default is \code{diag(ncol(x))}.}
 \item{log}{logical; if \code{TRUE}, densities d are given as log(d).}
 \item{method}{string specifying the matrix decomposition used to
   determine the matrix root of \code{sigma}.  Possible methods are
   eigenvalue decomposition (\code{"eigen"}, default),
   singular value decomposition (\code{"svd"}), and
   Cholesky decomposition (\code{"chol"}).  The
   Cholesky is typically fastest, not by much though.}
 \item{pre0.9_9994}{logical; if \code{FALSE}, the output produced in mvtnorm
                    versions up to 0.9-9993 is reproduced. In 0.9-9994, the
                    output is organized such that \code{rmvnorm(10,...)} has the
                    same first ten rows as \code{rmvnorm(100, ...)} when called
                    with the same seed.}
}
\author{Friedrich Leisch and Fabian Scheipl}
\seealso{\code{\link{pmvnorm}}, \code{\link{rnorm}}, \code{\link{qmvnorm}}}
\examples{
dmvnorm(x=c(0,0))
dmvnorm(x=c(0,0), mean=c(1,1))

sigma <- matrix(c(4,2,2,3), ncol=2)
x <- rmvnorm(n=500, mean=c(1,2), sigma=sigma)
colMeans(x)
var(x)

x <- rmvnorm(n=500, mean=c(1,2), sigma=sigma, method="chol")
colMeans(x)
var(x)

plot(x)
}
\keyword{distribution}
\keyword{multivariate}
