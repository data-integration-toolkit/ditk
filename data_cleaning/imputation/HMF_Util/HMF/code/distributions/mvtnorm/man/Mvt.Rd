\name{Mvt}
\alias{dmvt}
\alias{rmvt}
\title{The Multivariate t Distribution}
\description{
  These functions provide information about the multivariate \eqn{t}
  distribution with non-centrality parameter (or mode) \code{delta},
  scale matrix \code{sigma} and degrees of freedom \code{df}.
  \code{dmvt} gives the density and \code{rmvt}
  generates random deviates.
}
\usage{
rmvt(n, sigma = diag(2), df = 1, delta = rep(0, nrow(sigma)),
     type = c("shifted", "Kshirsagar"), ...)
dmvt(x, delta = rep(0, p), sigma = diag(p), df = 1, log = TRUE,
     type = "shifted")
}
\arguments{
  \item{x}{vector or matrix of quantiles. If \code{x} is a matrix, each
    row is taken to be a quantile.}
  \item{n}{number of observations.}
  \item{delta}{the vector of noncentrality parameters of length n, for
    \code{type = "shifted"} delta specifies the mode.}
  \item{sigma}{scale matrix, defaults to
    \code{diag(ncol(x))}.}
  \item{df}{degrees of freedom. \code{df = 0} or \code{df = Inf}
    corresponds to the multivariate normal distribution.}
  \item{log}{\code{\link{logical}} indicating whether densities \eqn{d}
    are given as \eqn{\log(d)}{log(d)}.}
  \item{type}{type of the noncentral multivariate \eqn{t} distribution.
    \code{type = "Kshirsagar"} corresponds
    to formula (1.4) in Genz and Bretz (2009) (see also
    Chapter 5.1 in Kotz and Nadarajah (2004)). This is the
    noncentral t-distribution  needed for calculating
    the power of multiple contrast tests under a normality
    assumption. \code{type = "shifted"} corresponds to the
    formula right before formula (1.4) in Genz and Bretz (2009)
    (see also formula (1.1) in Kotz and Nadarajah (2004)). It
    is a location shifted version of the central t-distribution.
    This noncentral multivariate \eqn{t} distribution appears for
    example as the Bayesian posterior distribution
    for the regression coefficients in a linear regression.
    In the central case both types coincide.
    Note that the defaults differ from the default
    in \code{\link{pmvt}()} (for reasons of backward
    compatibility).}
  \item{\dots}{additional arguments to \code{\link{rmvnorm}()},
    for example \code{method}.}
}
\details{
  If \eqn{\bm{X}}{X} denotes a random vector following a \eqn{t} distribution
  with location vector \eqn{\bm{0}}{0} and scale matrix
  \eqn{\Sigma}{Sigma} (written \eqn{X\sim t_\nu(\bm{0},\Sigma)}{X ~ t_nu(0,
    Sigma)}), the scale matrix (the argument
  \code{sigma}) is not equal to the covariance matrix \eqn{Cov(\bm{X})}{Cov(X)}
  of \eqn{\bm{X}}{X}. If the degrees of freedom \eqn{\nu}{nu} (the
  argument \code{df}) is larger than 2, then
  \eqn{Cov(\bm{X})=\Sigma\nu/(\nu-2)}{Cov(X)=Sigma
    nu/(nu-2)}. Furthermore,
  in this case the correlation matrix \eqn{Cor(\bm{X})}{Cor(X)} equals
  the correlation matrix corresponding to the scale matrix
  \eqn{\Sigma}{Sigma} (which can be computed with
  \code{\link{cov2cor}()}). Note that the scale matrix is sometimes
  referred to as \dQuote{dispersion matrix};
  see McNeil, Frey, Embrechts (2005, p. 74).

  For \code{type = "shifted"} the density
  \deqn{c(1+(x-\delta)'S^{-1}(x-\delta)/\nu)^{-(\nu+m)/2}}
  is implemented, where
  \deqn{c = \Gamma((\nu+m)/2)/((\pi \nu)^{m/2}\Gamma(\nu/2)|S|^{1/2}),}
  \eqn{S} is a positive definite symmetric matrix (the matrix
  \code{sigma} above), \eqn{\delta}{delta} is the
  non-centrality vector and \eqn{\nu}{nu} are the degrees of freedom.

  \code{df=0} historically leads to the multivariate normal
  distribution. From a mathematical point of view, rather
  \code{df=Inf} corresponds to the multivariate normal
  distribution. This is (now) also allowed for \code{rmvt()} and
  \code{dmvt()}.

  Note that \code{dmvt()} has default \code{log = TRUE}, whereas
  \code{\link{dmvnorm}()} has default \code{log = FALSE}.
}
\references{
  McNeil, A. J., Frey, R., and Embrechts, P. (2005).
  \emph{Quantitative Risk Management: Concepts, Techniques, Tools}.
  Princeton University Press.
}
\seealso{\code{\link{pmvt}()} and \code{\link{qmvt}()}}
\examples{
## basic evaluation
dmvt(x = c(0,0), sigma = diag(2))

## check behavior for df=0 and df=Inf
x <- c(1.23, 4.56)
mu <- 1:2
Sigma <- diag(2)
x0 <- dmvt(x, delta = mu, sigma = Sigma, df = 0) # default log = TRUE!
x8 <- dmvt(x, delta = mu, sigma = Sigma, df = Inf) # default log = TRUE!
xn <- dmvnorm(x, mean = mu, sigma = Sigma, log = TRUE)
stopifnot(identical(x0, x8), identical(x0, xn))

## X ~ t_3(0, diag(2))
x <- rmvt(100, sigma = diag(2), df = 3) # t_3(0, diag(2)) sample
plot(x)

## X ~ t_3(mu, Sigma)
n <- 1000
mu <- 1:2
Sigma <- matrix(c(4, 2, 2, 3), ncol=2)
set.seed(271)
x <- rep(mu, each=n) + rmvt(n, sigma=Sigma, df=3)
plot(x)

## Note that the call rmvt(n, mean=mu, sigma=Sigma, df=3) does *not*
## give a valid sample from t_3(mu, Sigma)! [and thus throws an error]
try(rmvt(n, mean=mu, sigma=Sigma, df=3))

## df=Inf correctly samples from a multivariate normal distribution
set.seed(271)
x <- rep(mu, each=n) + rmvt(n, sigma=Sigma, df=Inf)
set.seed(271)
x. <- rmvnorm(n, mean=mu, sigma=Sigma)
stopifnot(identical(x, x.))
}
\keyword{distribution}
\keyword{multivariate}
