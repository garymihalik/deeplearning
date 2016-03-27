#' Execution function that runs in the batch normalization mode
#'
#' @export


run_darch_bn <- function(darch,data){
  # helper function that repeat a row vector N times
  verticalize <- function(vector, N) {
    return(matrix(rep(vector, N), N, byrow = T))
  }


  darch <- resetExecOutput(darch)
  layers <- getLayers(darch)

  # If there's only one row of input data, convert vector to matrix
  # TODO make sure that data is matrix before passing it to this function
  if(is.null(dim(data))){
    data <- t(as.matrix(data))
  }

  numRows <- dim(data)[1]

  for(i in 1:length(layers)){
    ret <- getLayerWeights(darch, i)
    dimV_input <- dim(ret)[[1]] - 1
    dimV_output <- dim(ret)[[2]]

    weight <- ret[1:(dimV_input), ]
    beta <- verticalize(ret[(dimV_input + 1), ], numRows)

    if(length(getLayer(darch, i)) < 4) {
      gamma <- matrix(1, dimV_output)
    } else {
      gamma <- getLayer(darch, i)[[4]]
    }

    gamma <- verticalize(gamma, numRows)

    if(i < length(layers)){
      dropoutHidden <- darch@dropoutHidden
    } else {
      dropoutHidden <- 1
    }

    weight <- weight * dropoutHidden
    beta <- beta * dropoutHidden
    gamma <- gamma * dropoutHidden

    x <- data %*% weight

    if(length(getLayer(darch, i)) < 5) {
      mu <- colMeans(x)
    } else {
      mu <- getLayer(darch, i)[[5]]
    }

    mu <- verticalize(mu, numRows)

    if(length(getLayer(darch, i)) < 6) {
      sigma_2 <- colMeans(x^2) - mu[1,]^2
    } else {
      sigma_2 <- getLayer(darch, i)[[6]]
    }

    sigma_2 <- verticalize(sigma_2, numRows)

    ret <- batch_normalization(x, gamma, beta, mu, sigma_2)
    y <- ret[[4]]
    unit_matrix <- diag(dim(y)[[2]])
    ret <- layers[[i]][[2]](y, unit_matrix)
    data <- ret[[1]]
    darch <- addExecOutput(darch,data)
  }

  return(darch)
}
