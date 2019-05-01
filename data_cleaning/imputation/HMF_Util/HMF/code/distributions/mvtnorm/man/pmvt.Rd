\name{pmvt}
\alias{pmvt}
\title{ Multivariate t Distribution }
\description{

  Computes the the distribution function of the multivariate t distribution
  for arbitrary limits, degrees of freedom and correlation matrices
  based on algorithms by Genz and Bretz.

}
\usage{
pmvt(lower=-Inf, upper=Inf, delta=rep(0, length(lower)),
     df=1, corr=NULL, sigma=NULL, algorithm = GenzBretz(),
     type = c("Kshirsagar", "shifted"), ...)
}
\arguments{
  \item{lower}{ the vector of lower limits of length n.}
  \item{upper}{ the vector of upper limits of length n.}
  \item{delta}{ the vector of noncentrality parameters of length n, for
   \code{type = "shifted"} delta specifies the mode.}
  \item{df}{ degree of freedom as integer. Normal probabilities are computed for \code{df=0}.}
  \item{corr}{ the correlation matrix of dimension n.}
  \item{sigma}{ the scale matrix of dimension n. Either \code{corr} or
    \code{sigma} can be specified. If \code{sigma} is given, the
    problem is standardized. If neither \code{corr} nor
    \code{sigma} is given, the identity matrix is used
    for \code{sigma}. }
  \item{algorithm}{ an object of class \code{\link{GenzBretz}} or
    \code{\link{TVPACK}} defining the hyper parameters of this algorithm.}
  \item{type}{ type of the noncentral multivariate t distribution
              to be computed. \code{type = "Kshirsagar"} corresponds
              to formula (1.4) in Genz and Bretz (2009) (see also
	      Chapter 5.1 in Kotz and Nadarajah (2004)). This is the
	      noncentral t-distribution needed for calculating
	      the power of multiple contrast tests under a normality
	      assumption. \code{type = "shifted"} corresponds to the
	       formula right before formula (1.4) in Genz and Bretz (2009)
	      (see also formula (1.1) in Kotz and Nadarajah (2004)). It
	      is a location shifted version of the central
	      t-distribution. This noncentral multivariate t distribution appears for
	      example as the Bayesian posterior distribution
	      for the regression coefficients in a linear regression.
 	      In the central case both types coincide.
	    }
  \item{...}{additional parameters (currently given to \code{GenzBretz} for
              backward compatibility issues). }
}

\details{

This function involves the computation of central and noncentral
multivariate t-probabilities with arbitrary correlation matrices.
It involves both the computation of singular and nonsingular
probabilities.  The methodology (for default \code{algorithm =
  GenzBretz()}) is based on randomized quasi Monte Carlo methods and
described in Genz and Bretz (1999, 2002).
\cr
Because of the randomization, the result for this algorithm (slightly)
depends on \code{\link{.Random.seed}}.

For 2- and 3-dimensional problems one can also use the \code{\link{TVPACK}} routines
described by Genz (2004), which only handles semi-infinite integration
regions (and for \code{type = "Kshirsagar"} only central problems).

For \code{type = "Kshirsagar"} and a given correlation matrix
\code{corr}, for short \eqn{A}, say, (which has to be positive
semi-definite) and degrees of freedom \eqn{\nu} the following values are
numerically evaluated

%% FIXME add non-LaTeX alternative \deqn{LaTeX}{non-LaTex}
\deqn{I = 2^{1-\nu/2} / \Gamma(\nu/2) \int_0^\infty s^{\nu-1} \exp(-s^2/2) \Phi(s \cdot lower/\sqrt{\nu} - \delta,
  s \cdot upper/\sqrt{\nu} - \delta) \, ds }

where

\deqn{\Phi(a,b) = (det(A)(2\pi)^m)^{-1/2} \int_a^b \exp(-x^\prime Ax/2) \, dx}

is the multivariate normal distribution and \eqn{m} is the number of rows of
\eqn{A}.

For \code{type = "shifted"}, a positive definite symmetric matrix
\eqn{S} (which might be the correlation or the scale matrix),
mode (vector) \eqn{\delta} and degrees of freedom \eqn{\nu} the
following integral is evaluated:

\deqn{c\int_{lower_1}^{upper_1}...\int_{lower_m}^{upper_m}
  (1+(x-\delta)'S^{-1}(x-\delta)/\nu)^{-(\nu+m)/2}\, dx_1 ... dx_m,
}

where

\deqn{c = \Gamma((\nu+m)/2)/((\pi \nu)^{m/2}\Gamma(\nu/2)|S|^{1/2}),}

and \eqn{m} is the number of rows of \eqn{S}.

Note that both \code{-Inf} and \code{+Inf} may be specified in
the lower and upper integral limits in order to compute one-sided
probabilities.

Univariate problems are passed to \code{\link{pt}}.
If \code{df = 0}, normal probabilities are returned.

}

\value{
The evaluated distribution function is returned with attributes
  \item{error}{estimated absolute error and}
  \item{msg}{status message (a \code{\link{character}} string).}
}

\references{

Genz, A. and Bretz, F. (1999), Numerical computation of multivariate
t-probabilities with application to power calculation of multiple
contrasts. \emph{Journal of Statistical Computation and Simulation},
\bold{63}, 361--378.

Genz, A. and Bretz, F. (2002), Methods for the computation of multivariate
t-probabilities. \emph{Journal of Computational and Graphical Statistics},
\bold{11}, 950--971.

Genz, A. (2004), Numerical computation of rectangular bivariate and
trivariate normal and t-probabilities, \emph{Statistics and
Computing}, \bold{14}, 251--260.

Genz, A. and Bretz, F. (2009), \emph{Computation of Multivariate Normal and
t Probabilities}. Lecture Notes in Statistics, Vol. 195. Springer-Verlag,
Heidelberg.

S. Kotz and S. Nadarajah (2004), \emph{Multivariate t Distributions and
  Their Applications}. Cambridge University Press. Cambridge.

Edwards D. and Berry, Jack J. (1987), The efficiency of simulation-based
multiple comparisons. \emph{Biometrics}, \bold{43}, 913--928.

}

\source{
  \url{http://www.sci.wsu.edu/math/faculty/genz/homepage}
}

\seealso{\code{\link{qmvt}}}

\examples{

n <- 5
lower <- -1
upper <- 3
df <- 4
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
delta <- rep(0, 5)
prob <- pmvt(lower=lower, upper=upper, delta=delta, df=df, corr=corr)
print(prob)

pmvt(lower=-Inf, upper=3, df = 3, sigma = 1) == pt(3, 3)

# Example from R News paper (original by Edwards and Berry, 1987)

n <- c(26, 24, 20, 33, 32)
V <- diag(1/n)
df <- 130
C <- c(1,1,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,0,-1,-1,0,0,-1,0,0)
C <- matrix(C, ncol=5)
### scale matrix
cv <- C \%*\% V \%*\% t(C)
### correlation matrix
dv <- t(1/sqrt(diag(cv)))
cr <- cv * (t(dv) \%*\% dv)
delta <- rep(0,5)

myfct <- function(q, alpha) {
  lower <- rep(-q, ncol(cv))
  upper <- rep(q, ncol(cv))
  pmvt(lower=lower, upper=upper, delta=delta, df=df,
       corr=cr, abseps=0.0001) - alpha
}

### uniroot for this simple problem
round(uniroot(myfct, lower=1, upper=5, alpha=0.95)$root, 3)

# compare pmvt and pmvnorm for large df:

a <- pmvnorm(lower=-Inf, upper=1, mean=rep(0, 5), corr=diag(5))
b <- pmvt(lower=-Inf, upper=1, delta=rep(0, 5), df=rep(300,5),
          corr=diag(5))
a
b

stopifnot(round(a, 2) == round(b, 2))

# correlation and scale matrix

a <- pmvt(lower=-Inf, upper=2, delta=rep(0,5), df=3,
          sigma = diag(5)*2)
b <- pmvt(lower=-Inf, upper=2/sqrt(2), delta=rep(0,5),
          df=3, corr=diag(5))
attributes(a) <- NULL
attributes(b) <- NULL
a
b
stopifnot(all.equal(round(a,3) , round(b, 3)))

a <- pmvt(0, 1,df=10)
attributes(a) <- NULL
b <- pt(1, df=10) - pt(0, df=10)
stopifnot(all.equal(round(a,10) , round(b, 10)))

}

\keyword{distribution}
