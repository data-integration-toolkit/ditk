\name{rtmvnorm2}
\alias{rtmvnorm2}
\title{Sampling Random Numbers From The Truncated Multivariate Normal Distribution With Linear Constraints}
\description{
  This function generates random numbers from the truncated multivariate normal
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma} and general linear constraints 
  \deqn{lower \le D x \le upper}{lower <= D x <= upper} 
  with either rejection sampling or Gibbs sampling. 
}
\usage{
rtmvnorm2(n, mean = rep(0, nrow(sigma)), 
  sigma = diag(length(mean)), 
  lower = rep(-Inf, length = length(mean)), 
  upper = rep(Inf, length = length(mean)), 
  D = diag(length(mean)), 
  algorithm = c("gibbs", "gibbsR", "rejection"), ...)
}
\arguments{
  \item{n}{Number of random points to be sampled. Must be an integer \eqn{\ge 1}{>= 1}.}
  \item{mean}{Mean vector (d x 1), default is \code{rep(0, length = ncol(x))}.}
  \item{sigma}{Covariance matrix (d x d), default is \code{diag(ncol(x))}.}
  \item{lower}{Vector of lower truncation points (r x 1), 
              default is \code{rep( Inf, length = length(mean))}.}
  \item{upper}{Vector of upper truncation points (r x 1), 
              default is \code{rep( Inf, length = length(mean))}.}
  \item{D}{Matrix for linear constraints (r x d), defaults to diagonal matrix (d x d), i.e. r = d.}
  \item{algorithm}{Method used, possible methods are the Fortan Gibbs sampler ("gibbs", default), 
                  the Gibbs sampler implementation in R ("gibbsR") and
                  rejection sampling ("rejection")}
  \item{\dots}{additional parameters for Gibbs sampling, given to the internal method \code{rtmvnorm.gibbs()}, 
            such as \code{burn.in.samples}, \code{start.value} and \code{thinning}, see details in \code{\link{rtmvnorm}}}
   
}
\details{
This method allows for \eqn{r > d}{r > d} linear constraints, whereas \code{\link{rtmvnorm}}
requires a full-rank matrix D \eqn{(d \times d)}{(d x d)} and can only handle \eqn{r \le d}{r <= d} constraints at the moment.
The lower and upper bounds \code{lower} and \code{upper} are \eqn{(r \times 1)}{(r x 1)}, 
the matrix \code{D} is \eqn{(r \times d)}{(r x d)} and x is \eqn{(d \times 1)}{(d x 1)}. 
The default case is \eqn{r = d}{r = d} and \eqn{D = I_d}{D = I_d}.
}

\section{Warning}{This method will be merged with \code{\link{rtmvnorm}} in one of the next releases.}

\author{
Stefan Wilhelm
}

\seealso{
 \code{\link{rtmvnorm}}
}
\examples{
\dontrun{
################################################################################
#
# Example 5a: Number of linear constraints r > dimension d
#
################################################################################

# general linear restrictions a <= Dx <= b with x (d x 1); D (r x d); a,b (r x 1)

# Dimension d=2, r=3 linear constraints
#
# a1 <=    x1 + x2 <= b2
# a2 <=    x1 - x2 <= b2
# a3 <= 0.5x1 - x2 <= b3
#
# [ a1 ] <= [ 1     1 ] [ x1 ] <= [b1]
# [ a2 ]    [ 1    -1 ] [ x2 ]    [b2]
# [ a3 ]    [ 0.5  -1 ]           [b3]

D <- matrix(
      c(  1,  1,
          1, -1,
        0.5, -1), 3, 2, byrow=TRUE)
a <- c(0, 0, 0)
b <- c(1, 1, 1)

# mark linear constraints as lines
plot(NA, xlim=c(-0.5, 1.5), ylim=c(-1,1))
for (i in 1:3) {
  abline(a=a[i]/D[i, 2], b=-D[i,1]/D[i, 2], col="red")
  abline(a=b[i]/D[i, 2], b=-D[i,1]/D[i, 2], col="red")
}

### Gibbs sampling for general linear constraints a <= Dx <= b
mean <- c(0, 0)
sigma <- matrix(c(1.0, 0.2, 
                  0.2, 1.0), 2, 2)
x0 <- c(0.5, 0.2) # Gibbs sampler start value                  
X <- rtmvnorm2(n=1000, mean, sigma, lower=a, upper=b, D, start.value=x0)

# show random points within simplex
points(X, pch=20, col="black")
}
}

\keyword{distribution}
\keyword{multivariate}