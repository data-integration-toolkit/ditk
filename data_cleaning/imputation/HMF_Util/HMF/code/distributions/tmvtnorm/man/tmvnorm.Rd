% --- Source file: tmvtnorm.Rd ---
\name{tmvnorm}
\alias{dtmvnorm}
\title{Truncated Multivariate Normal Density}
\description{
  This function provides the joint density function for the truncated multivariate normal
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma}, lower and upper truncation points \code{lower} and \code{upper}.
  For convenience, it furthermore serves as a wrapper function for the one-dimensional and bivariate marginal densities 
  \code{dtmvnorm.marginal()} and \code{dtmvnorm.marginal2()} respectively when invoked with the \code{margin} argument.
}

\usage{
dtmvnorm(x, mean = rep(0, nrow(sigma)), 
  sigma = diag(length(mean)), 
  lower=rep(-Inf, length = length(mean)), 
  upper=rep( Inf, length = length(mean)),
  log=FALSE, 
  margin=NULL)
}

\arguments{
 \item{x}{Vector or matrix of quantiles. If \code{x} is a matrix, each row is taken to be a quantile.}
 \item{mean}{Mean vector, default is \code{rep(0, nrow(sigma))}.}
 \item{sigma}{Covariance matrix, default is \code{diag(length(mean))}.}
 \item{lower}{Vector of lower truncation points, 
              default is \code{rep(-Inf, length = length(mean))}.}
 \item{upper}{Vector of upper truncation points, 
              default is \code{rep( Inf, length = length(mean))}.}
 \item{log}{Logical; if \code{TRUE}, densities d are given as log(d).}
 \item{margin}{if \code{NULL} then the joint density is computed (the default), 
               if \code{MARGIN=1} then the one-dimensional marginal density in variate \code{q} (\code{q = 1..length(mean)}) is returned, 
               if \code{MARGIN=c(q,r)} then the bivariate marginal density in variates \code{q} and \code{r} 
               for \code{q,r = 1..length(mean)} and \eqn{q \ne r}{q != r} is returned.}
}

\details{

The computation of truncated multivariate normal probabilities and densities is done using conditional probabilities 
from the standard/untruncated multivariate normal distribution. 
So we refer to the documentation of the mvtnorm package and the methodology is described in
Genz (1992, 1993).

}
 
\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>}
\seealso{\code{\link{ptmvnorm}}, \code{\link[mvtnorm]{pmvnorm}}, \code{\link[mvtnorm]{rmvnorm}}, \code{\link[mvtnorm]{dmvnorm}},
         \code{\link{dtmvnorm.marginal}} and \code{\link{dtmvnorm.marginal2}} for marginal density functions}

\references{ 

Genz, A. (1992). Numerical computation of multivariate normal probabilities.
\emph{Journal of Computational and Graphical Statistics}, \bold{1}, 141--150 

Genz, A. (1993). Comparison of methods for the computation of multivariate
normal probabilities. \emph{Computing Science and Statistics}, \bold{25},
400--405

Johnson, N./Kotz, S. (1970). Distributions in Statistics: Continuous Multivariate Distributions
\emph{Wiley & Sons}, pp. 70--73

Horrace, W. (2005). Some Results on the Multivariate Truncated Normal Distribution. 
\emph{Journal of Multivariate Analysis}, \bold{94}, 209--221
}

\examples{
dtmvnorm(x=c(0,0), mean=c(1,1), upper=c(0,0))

###########################################
#
# Example 1: 
# truncated multivariate normal density        
#
############################################

x1<-seq(-2, 3, by=0.1)
x2<-seq(-2, 3, by=0.1)

density<-function(x)
{
  sigma=matrix(c(1, -0.5, -0.5, 1), 2, 2)
  z=dtmvnorm(x, mean=c(0,0), sigma=sigma, lower=c(-1,-1))
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

# compute density d for grid
d=fgrid(x1, x2, density)

# plot density as contourplot
contour(x1, x2, d, nlevels=5, main="Truncated Multivariate Normal Density", 
  xlab=expression(x[1]), ylab=expression(x[2]))
abline(v=-1, lty=3, lwd=2)
abline(h=-1, lty=3, lwd=2)

###########################################
#
# Example 2: 
# generation of random numbers
# from a truncated multivariate normal distribution        
#
############################################

sigma <- matrix(c(4,2,2,3), ncol=2)
x <- rtmvnorm(n=500, mean=c(1,2), sigma=sigma, upper=c(1,0))
plot(x, main="samples from truncated bivariate normal distribution",
  xlim=c(-6,6), ylim=c(-6,6), 
  xlab=expression(x[1]), ylab=expression(x[2]))
abline(v=1, lty=3, lwd=2, col="gray")
abline(h=0, lty=3, lwd=2, col="gray")
}
\keyword{distribution}
\keyword{multivariate}