\name{algorithms}
\alias{GenzBretz}
\alias{Miwa}
\alias{TVPACK}
\title{ Choice of Algorithm and Hyper Parameters }
\description{
  Choose between three algorithms for evaluating normal distributions
  and define hyper parameters.
}
\usage{
GenzBretz(maxpts = 25000, abseps = 0.001, releps = 0)
Miwa(steps = 128)
TVPACK(abseps = 1e-6)
}
\arguments{
  \item{maxpts}{maximum number of function values as integer.  The internal
    FORTRAN code always uses a minimum number depending on the dimension.
    (for example 752 for three-dimensional problems).}
  \item{abseps}{absolute error tolerance; for \code{TVPACK} only used
    for dimension 3.}
  \item{releps}{ relative error tolerance as double. }
  \item{steps}{ number of grid points to be evaluated. }
}
\details{

There are three algorithms available for evaluating normal
probabilities: The default is the randomized Quasi-Monte-Carlo procedure
by Genz (1992, 1993) and Genz and Bretz (2002) applicable to
arbitrary covariance structures and dimensions up to 1000.

For smaller dimensions (up to 20) and non-singular covariance matrices,
the algorithm by Miwa et al. (2003) can be used as well.

For two- and three-dimensional problems and semi-infinite integration
region, \code{TVPACK} implements an interface to the methods described
by Genz (2004).

}
\value{
    An object of class \code{GenzBretz} or \code{Miwa}
    defining hyper parameters.
}
\references{

Genz, A. (1992). Numerical computation of multivariate normal probabilities.
\emph{Journal of Computational and Graphical Statistics}, \bold{1}, 141--150.

Genz, A. (1993). Comparison of methods for the computation of multivariate
normal probabilities. \emph{Computing Science and Statistics}, \bold{25},
400--405.

Genz, A. and Bretz, F. (2002), Methods for the computation of multivariate
t-probabilities. \emph{Journal of Computational and Graphical Statistics},
\bold{11}, 950--971.

Genz, A. (2004), Numerical computation of rectangular bivariate and
trivariate normal and t-probabilities, \emph{Statistics and
Computing}, \bold{14}, 251--260.

Genz, A. and Bretz, F. (2009), \emph{Computation of Multivariate Normal and
t Probabilities}. Lecture Notes in Statistics, Vol. 195. Springer-Verlag,
Heidelberg.

Miwa, A., Hayter J. and Kuriki, S. (2003).
The evaluation of general non-centred orthant probabilities.
\emph{Journal of the Royal Statistical Society}, Ser. B, 65, 223--234.

Mi, X., Miwa, T. and Hothorn, T. (2009).
\code{mvtnorm}: New numerical algorithm for multivariate normal probabilities.
\emph{The R Journal} \bold{1}(1): 37--39.
\url{http://journal.r-project.org/archive/2009-1/RJournal_2009-1_Mi+et+al.pdf}
}
\keyword{distribution}
