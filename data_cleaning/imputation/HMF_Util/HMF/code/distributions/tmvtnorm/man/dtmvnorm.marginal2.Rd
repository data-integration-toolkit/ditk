\name{dtmvnorm.marginal2}
\Rdversion{1.1}
\alias{dtmvnorm.marginal2}

\title{
Bivariate marginal density functions from a Truncated Multivariate Normal distribution
}
\description{
 This function computes the bivariate marginal density function \eqn{f(x_q, x_r)} from a k-dimensional Truncated Multivariate Normal density function (k>=2).
 The bivariate marginal density is obtained by integrating out (k-2) dimensions as proposed by Tallis (1961).
 This function is basically an extraction of the Leppard and Tallis (1989) Fortran code for moments calculation,
 but extended to the double truncated case.
}

\usage{
dtmvnorm.marginal2(xq, xr, q, r, 
 mean = rep(0, nrow(sigma)), 
 sigma = diag(length(mean)), 
 lower = rep(-Inf, length = length(mean)), 
 upper = rep(Inf, length = length(mean)),
 log = FALSE, pmvnorm.algorithm=GenzBretz())
}

\arguments{
  \item{xq}{Value \eqn{x_q}}
  \item{xr}{Value \eqn{x_r}}
  \item{q}{Index position for \eqn{x_q} within mean vector to calculate the bivariate marginal density for.}
  \item{r}{Index position for \eqn{x_r} within mean vector to calculate the bivariate marginal density for.}
  \item{mean}{Mean vector, default is \code{rep(0, length = nrow(sigma))}.}
  \item{sigma}{Covariance matrix, default is \code{diag(length(mean))}.}
  \item{lower}{Vector of lower truncation points, 
   default is \code{rep(-Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points, 
   default is \code{rep( Inf, length = length(mean))}.}
  \item{log}{Logical; if \code{TRUE}, densities d are given as log(d).} 
  \item{pmvnorm.algorithm}{Algorithm used for \code{\link[mvtnorm]{pmvnorm}}}
}

\details{
  The bivariate marginal density function \eqn{f(x_q, x_r)} for \eqn{x \sim TN(\mu, \Sigma, a, b)} and \eqn{q \ne r} is defined as
  \deqn{F_{q,r}(x_q=c_q, x_r=c_r) = 
  \int^{b_1}_{a_1}...\int^{b_{q-1}}_{a_{q-1}}\int^{b_{q+1}}_{a_{q+1}}...\int^{b_{r-1}}_{a_{r-1}}\int^{b_{r+1}}_{a_{r+1}}...\int^{b_{k}}_{a_{k}} 
  \varphi{_{\alpha}}_{\Sigma}(x_s, c_q, c_r) dx_s}
}

\references{
Tallis, G. M. (1961). The moment generating function of the truncated multinormal distribution. 
\emph{Journal of the Royal Statistical Society, Series B}, \bold{23}, 223--229

Leppard, P. and Tallis, G. M. (1989). Evaluation of the Mean and Covariance of the Truncated Multinormal
\emph{Applied Statistics}, \bold{38}, 543--553

Manjunath B G and Wilhelm, S. (2009). Moments Calculation For the Double Truncated Multivariate Normal Density. 
Working Paper. Available at SSRN: \url{http://ssrn.com/abstract=1472153}
}
\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>, Manjunath B G <bgmanjunath@gmail.com>}

\examples{
  
  lower = c(-0.5, -1, -1)
  upper = c( 2.2,  2,  2)
  
  mean  = c(0,0,0)
  sigma = matrix(c(2.0, -0.6,  0.7, 
                  -0.6,  1.0, -0.2, 
                   0.7, -0.2,  1.0), 3, 3)
  
  # generate random samples from untruncated and truncated distribution
  Y = rmvnorm(10000, mean=mean, sigma=sigma)
  X = rtmvnorm(500,  mean=mean, sigma=sigma, lower=lower, upper=upper, 
      algorithm="gibbs")
    
  # compute bivariate marginal density of x1 and x2
  xq <- seq(lower[1], upper[1], by=0.1)
  xr <- seq(lower[2], upper[2], by=0.1)
  
  grid <- matrix(NA, length(xq), length(xr))
  for (i in 1:length(xq))
  {
    for (j in 1:length(xr))
    {
      grid[i,j] = dtmvnorm.marginal2(xq=xq[i], xr=xr[j], 
        q=1, r=2, sigma=sigma, lower=lower, upper=upper)
    }
  }
  
  plot(Y[,1], Y[,2], xlim=c(-4, 4), ylim=c(-4, 4), 
     main=expression("bivariate marginal density ("*x[1]*","*x[2]*")"), 
     xlab=expression(x[1]), ylab=expression(x[2]), col="gray80")
  points(X[,1], X[,2], col="black")
  
  lines(x=c(lower[1], upper[1], upper[1], lower[1], lower[1]), 
        y=c(lower[2],lower[2],upper[2],upper[2],lower[2]), 
        lty=2, col="red")
  contour(xq, xr, grid, add=TRUE, nlevels = 8, col="red", lwd=2)
  
  # scatterplot matrices for untruncated and truncated points
  require(lattice)
  splom(Y)
  splom(X)
}
\keyword{distribution}
\keyword{multivariate}
