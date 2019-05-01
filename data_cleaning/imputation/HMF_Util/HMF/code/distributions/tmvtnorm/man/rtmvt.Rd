\name{rtmvt}
\alias{rtmvt}
\title{Sampling Random Numbers From The Truncated Multivariate Student t Distribution}
\description{
  This function generates random numbers
  from the truncated multivariate Student-t
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma}, lower and upper truncation points \code{lower} and \code{upper}
  with either rejection sampling or Gibbs sampling.
}

\usage{
rtmvt(n, mean = rep(0, nrow(sigma)), sigma = diag(length(mean)), df = 1, 
  lower = rep(-Inf, length = length(mean)), 
  upper = rep(Inf, length = length(mean)),
  algorithm=c("rejection", "gibbs"), ...)
}

\arguments{
 \item{n}{Number of random points to be sampled. Must be an integer >= 1.}
 \item{mean}{Mean vector, default is \code{rep(0, length = ncol(x))}.}
 \item{sigma}{Covariance matrix, default is \code{diag(ncol(x))}.}
 \item{df}{Degrees of freedom parameter (positive, may be non-integer)}
 \item{lower}{Vector of lower truncation points,\\ 
              default is \code{rep(-Inf, length = length(mean))}.}
 \item{upper}{Vector of upper truncation points,\\ 
              default is \code{rep( Inf, length = length(mean))}.}
 \item{algorithm}{Method used, possible methods are rejection sampling ("rejection", default) and 
                  the R Gibbs sampler ("gibbs").}             
 \item{...}{additional parameters for Gibbs sampling, given to the internal method \code{rtmvt.gibbs()}, 
            such as \code{burn.in.samples}, \code{start.value} and \code{thinning}, see details}             
}

\details{
We sample \eqn{x \sim T(\mu, \Sigma, df)}{x ~ T(mean, Sigma, df)} subject to the rectangular truncation \eqn{lower \le x \le upper}{lower <= x <= upper}.
Currently, two random number generation methods are implemented: rejection sampling and the Gibbs Sampler.

For rejection sampling \code{algorithm="rejection"}, we sample from \code{\link[mvtnorm]{rmvt}}
and retain only samples inside the support region. The acceptance probability
will be calculated with \code{\link[mvtnorm]{pmvt}}. \code{\link[mvtnorm]{pmvt}} does only accept
integer degrees of freedom \code{df}. For non-integer \code{df}, \code{algorithm="rejection"}
will throw an error, so please use \code{algorithm="gibbs"} instead.  
 
The arguments to be passed along with \code{algorithm="gibbs"} are:
\describe{
\item{\code{burn.in.samples}}{number of samples in Gibbs sampling to be discarded as burn-in phase, must be non-negative.}
\item{\code{start.value}}{Start value (vector of length \code{length(mean)}) for the MCMC chain. If one is specified,
                    it must lie inside the support region (\eqn{lower \le start.value \le upper}{lower <= start.value <= upper}).
                    If none is specified, 
                    the start value is taken componentwise as the finite lower or upper boundaries respectively, 
                    or zero if both boundaries are infinite. Defaults to NULL.}
\item{\code{thinning}}{Thinning factor for reducing autocorrelation of random points in Gibbs sampling. Must be an integer \eqn{\ge 1}{>= 1}. 
                 We create a Markov chain of length \code{(n*thinning)} and take only those
                 samples \code{j=1:(n*thinning)} where \code{j \%\% thinning == 0}  
                 Defaults to 1 (no thinning of the chain).}
}
}

\section{Warning}{
The same warnings for the Gibbs sampler apply as for the method \code{\link{rtmvnorm}}.
}

\author{Stefan Wilhelm <Stefan.Wilhelm@financial.com>, Manjunath B G <bgmanjunath@gmail.com>}

\references{
Geweke, John F. (1991) Efficient Simulation from the Multivariate Normal and Student-t Distributions
	Subject to Linear Constraints.
\emph{Computer Science and Statistics. Proceedings of the 23rd Symposium on the Interface. Seattle Washington, April 21-24, 1991}, pp. 571--578
An earlier version of this paper is available at \url{http://www.biz.uiowa.edu/faculty/jgeweke/papers/paper47/paper47.pdf}
}

\examples{
###########################################################
#
# Example 1
#
###########################################################	

# Draw from multi-t distribution without truncation
X1 <- rtmvt(n=10000, mean=rep(0, 2), df=2)
X2 <- rtmvt(n=10000, mean=rep(0, 2), df=2, lower=c(-1,-1), upper=c(1,1))

###########################################################
#
# Example 2
#
###########################################################	

df = 2
mu = c(1,1,1)
sigma = matrix(c(  1, 0.5, 0.5,
                 0.5,   1, 0.5,
                 0.5, 0.5,   1), 3, 3)
lower = c(-2,-2,-2)
upper = c(2, 2, 2)

# Rejection sampling
X1 <- rtmvt(n=10000, mu, sigma, df, lower, upper)

# Gibbs sampling without thinning
X2 <- rtmvt(n=10000, mu, sigma, df, lower, upper, 
  algorithm="gibbs")

# Gibbs sampling with thinning
X3 <- rtmvt(n=10000, mu, sigma, df, lower, upper, 
  algorithm="gibbs", thinning=2)	
   
plot(density(X1[,1], from=lower[1], to=upper[1]), col="red", lwd=2,
  main="Gibbs vs. Rejection")
lines(density(X2[,1], from=lower[1], to=upper[1]), col="blue", lwd=2)
legend("topleft",legend=c("Rejection Sampling","Gibbs Sampling"), 
  col=c("red","blue"), lwd=2)

acf(X1)  # no autocorrelation in Rejection sampling
acf(X2)  # strong autocorrelation of Gibbs samples
acf(X3)  # reduced autocorrelation of Gibbs samples after thinning	
}

\keyword{distribution}
\keyword{multivariate}