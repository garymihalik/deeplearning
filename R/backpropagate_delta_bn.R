#' function that calculates the delta function of a darch object with batch
#' normalization
#'
#' Calculates the delta functions using backpropagation
#'
#' @export

backpropagate_delta_bn <- function(darch, trainData,targetData, errorFunc = meanSquareErr) {

  # Initiate the layers which have below elements:
  #   w
  #   x
  #   mu
  #   sigma_2
  #   x_hat
  #   y
  #   gamma
  #   beta
  #   output
  #   delta_w
  #   delta_x
  #   delta_mu
  #   delta_sigma_2
  #   delta_x_hat
  #   delta_y
  #   delta_gamma
  #   delta_beta
  #   derivative

  numLayers <- length(darch@layers)
  layers <- list()
  for(i in 1:numLayers) {
    layers[["weight"]][[i]] <- getLayer(darch, i)[[1]]
    layers[["x"]][[i]] <- c()
    layers[["mu"]][[i]] <- c()
    layers[["sigma_2"]][[i]] <- c()
    layers[["x_hat"]][[i]] <- c()
    layers[["y"]][[i]] <- c()
    layers[["beta"]][[i]] <- c()
    layers[["gamma"]][[i]] <- c()
    layers[["delta_weight"]][[i]] <- c()
    layers[["delta_x"]][[i]] <- c()
    layers[["delta_mu"]][[i]] <- c()
    layers[["delta_sigma_2"]][[i]] <- c()
    layers[["delta_x_hat"]][[i]] <- c()
    layers[["delta_y"]][[i]] <- c()
    layers[["delta_beta"]][[i]] <- list()
    layers[["delta_gamma"]][[i]] <- c()
    layers[["output"]][[i]] <- c()
    layers[["derivative"]][[i]] <- c()
  }

  stats <- getStats(darch)

  # apply input dropout mask to data
  # TODO same input dropout mask for all data in a batch?
  trainData <- applyDropoutMask(trainData, getDropoutMask(darch, 0))

  # 1. Forwardpropagate
  data <- trainData
  numRows <- dim(data)[1]
  for (i in 1:numLayers){
    data <- cbind(data,rep(1,numRows))
    weights <- layers[["weight"]][[i]]
    func <- getLayer(darch, i)[[2]]
    ret <- func(data,weights)

    # apply dropout masks to output, unless we're on the last layer
    if (i < numLayers)
    {
      ret[[1]] <- applyDropoutMask(ret[[1]], getDropoutMask(darch, i))
      ret[[2]] <- applyDropoutMask(ret[[2]], getDropoutMask(darch, i))
    }

    layers[["output"]][[i]] <- ret[[1]]
    data <- ret[[1]]
    layers[["derivative"]][[i]] <- ret[[2]]
  }
  rm(data,numRows)

  # 2. Calculate the Error on the network output
  errorDerivative <- errorFunc(layers[["output"]][[numLayers]], targetData)[[2]]

  layers[["delta_y"]][[numLayers]] <- errorDerivative * layers[["derivative"]][[numLayers]]

  if (numLayers > 1) {
    layers[["delta_weight"]][[numLayers]] <- t(layers[["output"]][[numLayers - 1]]) %*%
      layers[["delta_y"]][[numLayers]]
  } else {
    layers[["delta_weight"]][[numLayers]] <- t(trainData) %*%
      layers[["delta_y"]][[numLayers]]
  }

  layers[["delta_beta"]][[numLayers]] <- c(colSums(layers[["delta_y"]][[numLayers]]))

  # 3. Backpropagate the error
  for(i in (numLayers-1):1){
    weights <- layers[["weight"]][[i + 1]]
    # remove bias row
    weights <- weights[1:(nrow(weights)-1),,drop=F]

    error <-  layers[["delta_y"]][[i+1]] %*% t(weights)
    layers[["delta_y"]][[i]] <- error * layers[["derivative"]][[i]]
    if(i > 1) {
      layers[["delta_weight"]][[i]] <- t(layers[["output"]][[i-1]]) %*% layers[["delta_y"]][[i]]
    } else {
      layers[["delta_weight"]][[i]] <- t(trainData) %*% layers[["delta_y"]][[i]]
    }

    layers[["delta_beta"]][[i]] <- (colSums(layers[["delta_y"]][[i]]))
  }

  ret <- list()
  ret[[1]] <- layers[["delta_weight"]]
  ret[[2]] <- layers[["delta_beta"]]
  return(ret)
}
