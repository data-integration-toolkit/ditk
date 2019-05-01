% --- Source file: dtmvnorm-marginal.Rd ---
\encoding{latin1} 
\name{dtmvnorm.marginal}
\alias{dtmvnorm.marginal}

\title{One-dimensional marginal density functions from a Truncated Multivariate Normal distribution}
\description{
  This function computes the one-dimensional marginal density function from a Truncated Multivariate Normal density function
  using the algorithm given in Cartinhour (1990).
}

\usage{
dtmvnorm.marginal(xn, n=1, 
  mean= rep(0, nrow(sigma)), 
  sigma=diag(length(mean)), 
  lower=rep(-Inf, length = length(mean)), 
  upper=rep( Inf, length = length(mean)),
  log=FALSE)
}

\arguments{
 \item{xn}{Vector of quantiles to calculate the marginal density for.}
 \item{n}{Index position (1..k) within the random vector x to calculate the one-dimensional marginal density for.}
 \item{mean}{Mean vector, default is \code{rep(0, length = nrow(sigma))}.}
 \item{sigma}{Covariance matrix, default is \code{diag(length(mean))}.}
 \item{lower}{Vector of lower truncation points,\\ 
   default is \code{rep(-Inf, length = length(mean))}.}
 \item{upper}{Vector of upper truncation points,\\ 
   default is \code{rep( Inf, length = length(mean))}.}
 \item{log}{Logical; if \code{TRUE}, densities d are given as log(d).}  
}

\details{
  The one-dimensional marginal density \eqn{f_i(x_i)} of \eqn{x_i} is
  \deqn{f_i(x_i) = \int_{a_1}^{b_1} \ldots \int_{a_{i-1}}^{b_{i-1}} \int_{a_{i+1}}^{b_{i+1}} \ldots \int_{a_k}^{b_k} f(x) dx_{-i}}
  Note that the one-dimensional marginal density is not truncated normal, but only conditional densities are truncated normal.
}

\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>}

\references{
Cartinhour, J. (1990). One-dimensional marginal density functions of a truncated multivariate normal density function.
\emph{Communications in Statistics - Theory and Methods}, \bold{19}, 197--203

Arnold et al. (1993). The Nontruncated Marginal of a Truncated Bivariate Normal Distribution.
\emph{Psychometrika}, \bold{58}, 471--488
}

\examples{
#############################################
#
# Example 1: truncated bivariate normal
#
#############################################

# parameters of the bivariate normal distribution
sigma = matrix(c(1   , 0.95,
                 0.95, 1    ), 2, 2)
mu = c(0,0)

# sample from multivariate normal distribution
X = rmvnorm(5000, mu, sigma)

# tuncation in x2 with x2 <= 0
X.trunc = X[X[,2]<0,]

# plot the realisations before and after truncation
par(mfrow=c(2,2))
plot(X, col="gray", xlab=expression(x[1]), ylab=expression(x[2]), 
  main="realisations from a\n truncated bivariate normal distribution")
points(X.trunc)
abline(h=0, lty=2, col="gray")
#legend("topleft", col=c("gray", "black")

# marginal density for x1 from realisations
plot(density(X.trunc[,1]), main=expression("marginal density for "*x[1]))

# one-dimensional marginal density for x1 using the formula
x <- seq(-5, 5, by=0.01)
fx <- dtmvnorm.marginal(x, n=1, mean=mu, sigma=sigma, 
  lower=c(-Inf,-Inf), upper=c(Inf,0))
lines(x, fx, lwd=2, col="red")

# marginal density for x2
plot(density(X.trunc[,2]), main=expression("marginal density for "*x[2]))

# one-dimensional marginal density for x2 using the formula
x <- seq(-5, 5, by=0.01)
fx <- dtmvnorm.marginal(x, n=2, mean=mu, sigma=sigma, 
  lower=c(-Inf,-Inf), upper=c(Inf,0))
lines(x, fx, lwd=2, col="blue")

#############################################
#
# Example 2 : truncated trivariate normal
#
#############################################

# parameters of the trivariate normal distribution
sigma = outer(1:3,1:3,pmin)
mu    = c(0,0,0)

# sample from multivariate normal distribution
X     = rmvnorm(2000, mu, sigma)

# truncation in x2 and x3 : x2 <= 0, x3 <= 0
X.trunc = X[X[,2]<=0 & X[,3]<=0,]

par(mfrow=c(2,3))
plot(X, col="gray", xlab=expression(x[1]), ylab=expression(x[2]), 
  main="realisations from a\n truncated trivariate normal distribution")
points(X.trunc, col="black")
abline(h=0, lty=2, col="gray")

plot(X[,2:3], col="gray", xlab=expression(x[2]), ylab=expression(x[3]), 
  main="realisations from a\n truncated trivariate normal distribution")
points(X.trunc[,2:3], col="black")
abline(h=0, lty=2, col="gray")
abline(v=0, lty=2, col="gray")

plot(X[,c(1,3)], col="gray", xlab=expression(x[1]), ylab=expression(x[3]), 
  main="realisations from a\n truncated trivariate normal distribution")
points(X.trunc[,c(1,3)], col="black")
abline(h=0, lty=2, col="gray")

# one-dimensional marginal density for x1 from realisations and formula
plot(density(X.trunc[,1]), main=expression("marginal density for "*x[1]))
x <- seq(-5, 5, by=0.01)
fx <- dtmvnorm.marginal(x, n=1, mean=mu, sigma=sigma, 
  lower=c(-Inf,-Inf,-Inf), upper=c(Inf,0,0))
lines(x, fx, lwd=2, col="red")

# one-dimensional marginal density for x2 from realisations and formula
plot(density(X.trunc[,2]), main=expression("marginal density for "*x[2]))
x <- seq(-5, 5, by=0.01)
fx <- dtmvnorm.marginal(x, n=2, mean=mu, sigma=sigma, 
  lower=c(-Inf,-Inf,-Inf), upper=c(Inf,0,0))
lines(x, fx, lwd=2, col="red")

# one-dimensional marginal density for x3 from realisations and formula
plot(density(X.trunc[,3]), main=expression("marginal density for "*x[3]))
x <- seq(-5, 5, by=0.01)
fx <- dtmvnorm.marginal(x, n=3, mean=mu, sigma=sigma, 
  lower=c(-Inf,-Inf,-Inf), upper=c(Inf,0,0))
lines(x, fx, lwd=2, col="red")
}

\keyword{distribution}
\keyword{multivariate}
