
# Assume a logit or softmax output layer
#'
#'
#'
#' @export

crossEntropyErr <- function(output, target) {
  # err <- - sum(target[] * log(output[]) + (1 - target[]) * log(1 - output[]))
  err <- - sum(target * log(output) + (1 - target) * log(1 - output))
  err2 <- (1-target)/(1-output) - target/output
  ret <- list()
  ret[[1]] <- err
  ret[[2]] <- err2
  ret[[3]] <- "Cross Entropy Error"
  return(ret)
}

#' @export

meanSquareErr <- function(output, target) {
  err <- 1/2 * sum(output - target)^2 / dim(output)[[1]]
  err2 <-  (output - target)
  ret <- list()
  ret[[1]] <- err
  ret[[2]] <- err2
  ret[[3]] <- "Mean Squared Error"
  return(ret)

}
