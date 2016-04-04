#' Resets the mu and sigmas of a darch instance to 0 and 1
#'
#' Resets the mu and sigmas of a darch instance
#'
#' @export

reset_population_mu_sigma <- function (darch) {
  numLayers <- length(darch@layers)
  epsilon <- exp(-12) # a numerical stablizaer used in batch normalization

  for(i in 1:numLayers) {
    ret <- getLayer(darch, i)[[1]]
    dimV_output <- dim(ret)[[2]]
    darch@layers[[i]][[5]] <- rep(0, dimV_output)
    darch@layers[[i]][[6]] <- rep(1 - epsilon, dimV_output)
  }

  return (darch)
}
