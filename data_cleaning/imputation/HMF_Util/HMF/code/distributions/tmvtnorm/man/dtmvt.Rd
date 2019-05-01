\name{dtmvt}
\alias{dtmvt}
\title{Truncated Multivariate Student t Density}
\description{
This function provides the joint density function for the truncated multivariate Student t
  distribution with mean vector equal to \code{mean}, covariance matrix
  \code{sigma}, degrees of freedom parameter \code{df} and 
  lower and upper truncation points \code{lower} and \code{upper}.
}
\usage{
dtmvt(x, mean = rep(0, nrow(sigma)), sigma = diag(length(mean)), df = 1, 
lower = rep(-Inf, length = length(mean)), 
upper = rep(Inf, length = length(mean)), log = FALSE)
}

\arguments{
\item{x}{Vector or matrix of quantiles. If \code{x} is a matrix, each row is taken to be a quantile.}
\item{mean}{Mean vector, default is \code{rep(0, nrow(sigma))}.}
\item{sigma}{Covariance matrix, default is \code{diag(length(mean))}.}
\item{df}{degrees of freedom parameter}
\item{lower}{Vector of lower truncation points, 
              default is \code{rep(-Inf, length = length(mean))}.}
\item{upper}{Vector of upper truncation points, 
              default is \code{rep( Inf, length = length(mean))}.}
\item{log}{Logical; if \code{TRUE}, densities d are given as log(d).}
}

\details{
The Truncated Multivariate Student t Distribution is a conditional Multivariate Student t distribution 
subject to (linear) constraints \eqn{a \le \bold{x} \le b}.

The density of the \eqn{p}-variate Multivariate Student t distribution with \eqn{\nu}{nu} degrees of freedom is 
\deqn{
 f(\bold{x}) = \frac{\Gamma((\nu + p)/2)}{(\pi\nu)^{p/2} \Gamma(\nu/2) \|\Sigma\|^{1/2}}
               [ 1 + \frac{1}{\nu} (x - \mu)^T \Sigma^{-1} (x - \mu) ]^{- (\nu + p) / 2}
}
The density of the truncated distribution \eqn{f_{a,b}(x)} with constraints \eqn{(a \le x \le b)}{a <= x <= b} is accordingly
\deqn{
 f_{a,b}(x) = \frac{f(\bold{x})} {P(a \le x \le b)}
}
}

\value{
 a numeric vector with density values
}

\seealso{
  \code{\link{ptmvt}} and \code{\link{rtmvt}} for probabilities and random number generation in the truncated case, 
  see \code{\link[mvtnorm]{dmvt}}, \code{\link[mvtnorm]{rmvt}} and \code{\link[mvtnorm]{pmvt}} 
  for the untruncated multi-t distribution.
}

\references{
Geweke, J. F. (1991) Efficient simulation from the multivariate normal and Student-t distributions 
 subject to linear constraints and the evaluation of constraint probabilities.
 \url{http://www.biz.uiowa.edu/faculty/jgeweke/papers/paper47/paper47.pdf} 
 
Samuel Kotz, Saralees Nadarajah (2004). Multivariate t Distributions and Their Applications. 
  \emph{Cambridge University Press}
}
  
\author{Stefan Wilhelm \email{wilhelm@financial.com}}

\examples{
# Example

x1 <- seq(-2, 3, by=0.1)
x2 <- seq(-2, 3, by=0.1)

mean <- c(0,0)
sigma <- matrix(c(1, -0.5, -0.5, 1), 2, 2)
lower <- c(-1,-1)


density <- function(x)
{
	z=dtmvt(x, mean=mean, sigma=sigma, lower=lower)
	z
}

fgrid <- function(x, y, f)
{
	z <- matrix(nrow=length(x), ncol=length(y))
	for(m in 1:length(x)){
		for(n in 1:length(y)){
			z[m,n] <- f(c(x[m], y[n]))
		}
	}
	z
}

# compute multivariate-t density d for grid
d <- fgrid(x1, x2, function(x) dtmvt(x, mean=mean, sigma=sigma, lower=lower))

# compute multivariate normal density d for grid
d2 <- fgrid(x1, x2, function(x) dtmvnorm(x, mean=mean, sigma=sigma, lower=lower))

# plot density as contourplot
contour(x1, x2, d, nlevels=5, main="Truncated Multivariate t Density", 
		xlab=expression(x[1]), ylab=expression(x[2]))

contour(x1, x2, d2, nlevels=5, add=TRUE, col="red")
abline(v=-1, lty=3, lwd=2)
abline(h=-1, lty=3, lwd=2)
}
\keyword{distribution}
\keyword{multivariate}