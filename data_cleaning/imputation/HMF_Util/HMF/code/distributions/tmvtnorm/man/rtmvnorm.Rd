\name{rtmvnorm}
\alias{rtmvnorm}
\alias{rtmvnorm.sparseMatrix}
\title{Sampling Random Numbers From The Truncated Multivariate Normal Distribution}
\description{
  This function generates random numbers
  from the truncated multivariate normal
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma} (or alternatively precision matrix \code{H}), 
  lower and upper truncation points \code{lower} and \code{upper}
  with either rejection sampling or Gibbs sampling.
}
\usage{
rtmvnorm(n, mean = rep(0, nrow(sigma)), 
  sigma = diag(length(mean)),
  lower=rep(-Inf, length = length(mean)), 
  upper=rep( Inf, length = length(mean)),
  D = diag(length(mean)),
  H = NULL, 
  algorithm=c("rejection", "gibbs", "gibbsR"),
  ...)
  
rtmvnorm.sparseMatrix(n, mean = rep(0, nrow(H)), 
    H = sparseMatrix(i=1:length(mean), j=1:length(mean), x=1),
    lower = rep(-Inf, length = length(mean)), 
    upper = rep( Inf, length = length(mean)),
    ...)  
}

\arguments{
 \item{n}{Number of random points to be sampled. Must be an integer \eqn{\ge 1}{>= 1}.}
 \item{mean}{Mean vector, default is \code{rep(0, length = ncol(x))}.}
 \item{sigma}{Covariance matrix, default is \code{diag(ncol(x))}.}
 \item{lower}{Vector of lower truncation points, 
              default is \code{rep(-Inf, length = length(mean))}.}
 \item{upper}{Vector of upper truncation points, 
              default is \code{rep( Inf, length = length(mean))}.}
 \item{D}{Matrix for linear constraints, defaults to diagonal matrix.}
 \item{H}{Precision matrix, default is \code{NULL}.}              
 \item{algorithm}{Method used, possible methods are rejection sampling ("rejection", default), 
                  the Fortan Gibbs sampler ("gibbs") 
                  and the old Gibbs sampler implementation in R ("gibbsR").}
 \item{...}{additional parameters for Gibbs sampling, given to the internal method \code{rtmvnorm.gibbs()}, 
            such as \code{burn.in.samples}, \code{start.value} and \code{thinning}, see details}
}

\details{

The generation of random numbers from a truncated multivariate normal distribution is done 
using either rejection sampling or Gibbs sampling.

\bold{Rejection sampling}\cr
Rejection sampling is done from the standard multivariate normal distribution. 
So we use the function \code{\link[mvtnorm]{rmvnorm}} of the \pkg{mvtnorm} package to generate
proposals which are either accepted if they are inside the support region or rejected.
In order to speed up the generation of N samples from the truncated distribution, 
we first calculate the acceptance rate alpha from the truncation points and then generate N/alpha samples iteratively 
until we have got N samples. This typically does not take more than 2-3 iterations.
Rejection sampling may be very inefficient when the support region is small (i.e. in higher dimensions) 
which results in very low acceptance rates alpha. In this case the Gibbs sampler is preferable.

\bold{Gibbs sampling}\cr
The Gibbs sampler samples from univariate conditional distributions, 
so all samples can be accepted except for a burn-in period. 
The number of burn-in samples to be discarded can be specified, as well as a start value of the chain.
If no start value is given, we determine a start value from the support region
using either lower bound or upper bound if they are finite, or 0 otherwise.

The Gibbs sampler has been reimplemented in Fortran 90 for performance reasons (\code{algorithm="gibbs"}). 
The old R implementation is still accessible through \code{algorithm="gibbsR"}.

The arguments to be passed along with \code{algorithm="gibbs"} or \code{algorithm="gibbsR"} are:
\describe{
\item{\code{burn.in.samples}}{number of samples in Gibbs sampling to be discarded as burn-in phase, must be non-negative.}
\item{\code{start.value}}{Start value (vector of length \code{length(mean)}) for the MCMC chain. If one is specified,
                    it must lie inside the support region (\eqn{lower <= start.value <= upper}).
                    If none is specified, 
                    the start value is taken componentwise as the finite lower or upper boundaries respectively, 
                    or zero if both boundaries are infinite. Defaults to NULL.}
\item{\code{thinning}}{Thinning factor for reducing autocorrelation of random points in Gibbs sampling. Must be an integer >= 1. 
                 We create a Markov chain of length \code{(n*thinning)} and take only those
                 samples \code{j=1:(n*thinning)} where \code{j \%\% thinning == 0}  
                 Defaults to 1 (no thinning of the chain).}
}

\bold{Sampling with linear constraints}\cr
We extended the method to also simulate from a multivariate normal distribution
subject to general linear constraints  \eqn{lower <= D x <= upper}. 
For general D, both rejection sampling or Gibbs sampling according to Geweke (1991)
are available.

\bold{Gibbs sampler and the use of the precision matrix H}\cr
Why is it important to have a random sampler that works with the precision matrix?
Especially in Bayesian and spatial statistics, there are a number of high-dimensional 
applications where the precision matrix \code{H} is readily available, 
but is sometimes nearly singular and cannot be easily inverted to sigma.
Additionally, it turns out that the Gibbs sampler formulas are much simpler
in terms of the precision matrix than in terms of the covariance matrix. 
See the details of the Gibbs sampler implementation in the package vignette or for example Geweke (2005), pp.171-172.
(Thanks to Miguel Godinho de Matos from Carnegie Mellon University for pointing me to this.)
Therefore, we now provide an interface for the direct use of the precision matrix \code{H}
in \code{rtmvnorm()}.

\bold{Gibbs sampler with sparse precision matrix H}\cr
The size of the covariance matrix \code{sigma} or precision matrix \code{H} 
- if expressed as a dense \code{\link[base]{matrix}} - grows quadratic with the number of dimensions d.
For high-dimensional problems (such as d > 5000), 
it is no longer efficient and appropriate to work with dense matrix representations, 
as one quickly runs into memory problems.\cr 
It is interesting to note that in many applications the precision matrix, 
which holds the conditional dependencies, will be sparse, whereas the covariance matrix 
will be dense. 
Hence, expressing H as a sparse matrix will significantly reduce the amount of memory to store this matrix 
and allows much larger problems to be handled. 
In the current version of the package, the precision matrix (not \code{sigma} since it will be dense in most cases)
can be passed to \code{rtmvnorm.sparseMatrix()} as a \code{\link[Matrix]{sparseMatrix}} from the \code{Matrix} package.
See the examples section below for a usage example.
}

\section{Warning}{
A word of caution is needed for useRs that are not familiar with Markov Chain Monte Carlo methods like Gibbs sampling:

Rejection sampling is exact in the sense that we are sampling directly from the target distribution 
and the random samples generated are independent. So it is clearly the default
method.

Markov Chain Monte Carlo methods are only approximate methods, which may suffer from several problems:
\itemize{
\item{Poor mixing}
\item{Convergence problems}
\item{Correlation among samples}
}
Diagnostic checks for Markov Chain Monte Carlo 
include trace plots, CUSUM plots and autocorrelation plots like \code{\link{acf}}. For
a survey see for instance Cowles (1996).

That is, consecutive samples generated from \code{rtmvnorm(..., algorithm=c("gibbs", "gibbsR"))} are correlated (see also example 3 below).
One way of reducing the autocorrelation among the random samples is "thinning" the Markov chain, that is
recording only a subset/subsequence of the chain. For example, one could record only every 100th sample,
which clearly reduces the autocorrelation and "increases the independence". 
But thinning comes at the cost of higher computation times, since the chain has to run much longer.
We refer to autocorrelation plots in order to determine optimal thinning.
}
 
\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>, Manjunath B G <bgmanjunath@gmail.com>}
\seealso{\code{\link{ptmvnorm}}, \code{\link[mvtnorm]{pmvnorm}}, \code{\link[mvtnorm]{rmvnorm}}, \code{\link[mvtnorm]{dmvnorm}}}

\references{

Alan Genz, Frank Bretz, Tetsuhisa Miwa, Xuefei Mi, Friedrich Leisch, Fabian Scheipl, Torsten Hothorn (2009). mvtnorm:
  Multivariate Normal and t Distributions. R package version 0.9-7. URL \url{http://CRAN.R-project.org/package=mvtnorm} 

Johnson, N./Kotz, S. (1970). Distributions in Statistics: Continuous Multivariate Distributions
\emph{Wiley & Sons}, pp. 70--73

Horrace, W. (2005). Some Results on the Multivariate Truncated Normal Distribution. 
\emph{Journal of Multivariate Analysis}, \bold{94}, 209--221

Jayesh H. Kotecha and Petar M. Djuric (1999). Gibbs Sampling Approach For Generation of Truncated Multivariate Gaussian Random Variables
\emph{IEEE Computer Society}, 1757--1760

Cowles, M. and Carlin, B. (1996). Markov Chain Monte Carlo Convergence Diagnostics: A Comparative Review
\emph{Journal of the American Statistical Association}, \bold{91}, 883--904

Geweke, J. F. (1991). Effcient Simulation from the Multivariate Normal and Student-t Distributions 
Subject to Linear Constraints 
\emph{Computer Science and Statistics. Proceedings of the 23rd Symposium on the Interface. Seattle Washington, April 21-24, 1991}, 571--578

Geweke, J. F. (2005). Contemporary Bayesian Econometrics and Statistics, \emph{Wiley & Sons}, pp.171--172
}

\examples{
################################################################################
#
# Example 1: 
# rejection sampling in 2 dimensions       
#
################################################################################

sigma <- matrix(c(4,2,2,3), ncol=2)
x <- rtmvnorm(n=500, mean=c(1,2), sigma=sigma, upper=c(1,0))
plot(x, main="samples from truncated bivariate normal distribution",
  xlim=c(-6,6), ylim=c(-6,6), 
  xlab=expression(x[1]), ylab=expression(x[2]))
abline(v=1, lty=3, lwd=2, col="gray")
abline(h=0, lty=3, lwd=2, col="gray")

################################################################################
#
# Example 2: 
# Gibbs sampler for 4 dimensions
#
################################################################################

C <- matrix(0.8, 4, 4)
diag(C) <- rep(1, 4)
lower <- rep(-4, 4)
upper <- rep(-1, 4)

# acceptance rate alpha
alpha <- pmvnorm(lower=lower, upper=upper, mean=rep(0,4), sigma=C)
alpha

# Gibbs sampler
X1 <- rtmvnorm(n=20000, mean = rep(0,4), sigma=C, lower=lower, upper=upper, 
  algorithm="gibbs", burn.in.samples=100)
# Rejection sampling
X2 <- rtmvnorm(n=5000, mean = rep(0,4), sigma=C, lower=lower, upper=upper)

colMeans(X1)
colMeans(X2)

plot(density(X1[,1], from=lower[1], to=upper[1]), col="red", lwd=2, 
     main="Kernel density estimates from random samples 
     generated by Gibbs vs. Rejection sampling")
lines(density(X2[,1], from=lower[1], to=upper[1]), col="blue", lwd=2)
legend("topleft",legend=c("Gibbs Sampling","Rejection Sampling"), 
  col=c("red","blue"), lwd=2, bty="n")

################################################################################
#
# Example 3: 
# Autocorrelation plot for Gibbs sampler
# with and without thinning
#
################################################################################

sigma <- matrix(c(4,2,2,3), ncol=2)
X1 <- rtmvnorm(n=10000, mean=c(1,2), sigma=sigma, upper=c(1,0), 
  algorithm="rejection")
acf(X1)
# no autocorrelation among random points

X2 <- rtmvnorm(n=10000, mean=c(1,2), sigma=sigma, upper=c(1,0), 
  algorithm="gibbs")
acf(X2)
# exhibits autocorrelation among random points

X3 <- rtmvnorm(n=10000, mean=c(1,2), sigma=sigma, upper=c(1,0), 
  algorithm="gibbs", thinning=2)
acf(X3)
# reduced autocorrelation among random points

plot(density(X1[,1], to=1))
lines(density(X2[,1], to=1), col="blue")
lines(density(X3[,1], to=1), col="red")

################################################################################
#
# Example 4: Univariate case
#
################################################################################

X <- rtmvnorm(100, mean=0, sigma=1, lower=-1, upper=1)

################################################################################
#
# Example 5: Linear Constraints
#
################################################################################

mean  <- c(0, 0)
sigma <- matrix(c(10, 0,
                   0, 1), 2, 2)

# Linear Constraints
#
# a1 <= x1 + x2 <= b2
# a2 <= x1 - x2 <= b2
#
# [ a1 ] <= [ 1   1 ] [ x1 ] <= [b1]
# [ a2 ]    [ 1  -1 ] [ x2 ]    [b2]
a     <- c(-2, -2)
b     <- c( 2,  2)
D     <- matrix(c(1, 1,
                  1, -1), 2, 2)                   

X <- rtmvnorm(n=10000, mean, sigma, lower=a, upper=b, D=D, algorithm="gibbsR")
plot(X, main="Gibbs sampling for multivariate normal 
              with linear constraints according to Geweke (1991)")

# mark linear constraints as lines
for (i in 1:nrow(D)) {
  abline(a=a[i]/D[i, 2], b=-D[i,1]/D[i, 2], col="red")
  abline(a=b[i]/D[i, 2], b=-D[i,1]/D[i, 2], col="red")
}
                          
################################################################################
#
# Example 6: Using precision matrix H rather than sigma
#
################################################################################

lower <- c(-1, -1)
upper <- c(1, 1)
mean <- c(0.5, 0.5)
sigma <- matrix(c(1, 0.8, 0.8, 1), 2, 2)
H <- solve(sigma)
D <- matrix(c(1, 1, 1, -1), 2, 2)
X <- rtmvnorm(n=1000, mean=mean, H=H, lower=lower, upper=upper, D=D, algorithm="gibbs")
plot(X, main="Gibbs sampling with precision matrix and linear constraints")

################################################################################
#
# Example 7: Using sparse precision matrix H in high dimensions
#
################################################################################

\dontrun{
d <- 1000
I_d <- sparseMatrix(i=1:d, j=1:d, x=1)
W <- sparseMatrix(i=c(1:d, 1:(d-1)), j=c(1:d, (2:d)), x=0.5)
H <- t(I_d - 0.5 * W) %*% (I_d - 0.5 * W)
lower <- rep(0, d)
upper <- rep(2, d)

# Gibbs sampler generates n=100 draws in d=1000 dimensions
X <- rtmvnorm.sparseMatrix(n=100, mean = rep(0,d), H=H, lower=lower, upper=upper,
  burn.in.samples=100)
colMeans(X)  
cov(X)
}
}
\keyword{distribution}
\keyword{multivariate}