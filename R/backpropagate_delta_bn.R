#' function that calculates the delta function of a darch object with batch
#' normalization
#'
#' Calculates the delta functions using backpropagation
#'
#' @export

backpropagate_delta_bn <- function(darch, trainData,targetData, errorFunc = meanSquareErr) {

  # helper function that repeat a row vector N times
  verticalize <- function(vector, N) {
    return(matrix(rep(vector, N), N, byrow = T))
  }

  numLayers <- length(darch@layers)
  layers <- list()
  epsilon <- exp(-12) # a numerical stablizaer used in batch normalization
  numObs <- dim(trainData)[[1]]

  for(i in 1:numLayers) {
    ret <- getLayer(darch, i)[[1]]
    dimV_input <- dim(ret)[[1]] - 1
    dimV_output <- dim(ret)[[2]]

    layers[["weight"]][[i]] <- ret[1:dimV_input, ]

    if(length(getLayer(darch, i)) < 4) {
      layers[["gamma"]][[i]] <-
        matrix(rep(1, dimV_output * numObs), numObs, byrow = T)
    } else {
      layers[["gamma"]][[i]] <-
        matrix(rep(getLayer(darch, i)[[4]], numObs), numObs, byrow = T)
    }

    layers[["beta"]][[i]] <- verticalize(ret[(dimV_input + 1),], numObs)

    layers[["x"]][[i]] <- list()
    layers[["mu"]][[i]] <- list()
    layers[["sigma_2"]][[i]] <- list()
    layers[["x_hat"]][[i]] <- list()
    layers[["y"]][[i]] <- list()

    layers[["delta_weight"]][[i]] <- list()
    layers[["delta_x"]][[i]] <- list()
    layers[["delta_y"]][[i]] <- list()
    layers[["delta_beta"]][[i]] <- list()
    layers[["delta_gamma"]][[i]] <- list()
    layers[["output"]][[i]] <- list()
    layers[["derivative"]][[i]] <- list()
  }

  # apply input dropout mask to data
  # TODO same input dropout mask for all data in a batch?
  trainData <- applyDropoutMask(trainData, getDropoutMask(darch, 0))

  # 1. Forwardpropagate
  data <- trainData
  for (i in 1:numLayers){
    weights <- layers[["weight"]][[i]]
    func <- getLayer(darch, i)[[2]]
    # Batch Normalization
    layers[["x"]][[i]] <- data %*% weights
    ret <- batch_normalization(layers[["x"]][[i]],
                               layers[["gamma"]][[i]],
                               layers[["beta"]][[i]],
                               epsilon)
    layers[["mu"]][[i]] <- ret[[1]]
    layers[["sigma_2"]][[i]] <- ret[[2]]
    layers[["x_hat"]][[i]] <- ret[[3]]
    layers[["y"]][[i]] <- ret[[4]]

    ret <- list()

    unit_matrix <- diag(dim(layers[['y']][[i]])[[2]])
    ret <- func(layers[["y"]][[i]],unit_matrix)
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

  # 2. Calculate the Error on the network output
  errorDerivative <- errorFunc(layers[["output"]][[numLayers]], targetData)[[2]]
  layers[["delta_y"]][[numLayers]] <- errorDerivative * layers[["derivative"]][[numLayers]]

  ret <- batch_normalization_differential(layers[["delta_y"]][[numLayers]],
                                          layers[["mu"]][[numLayers]],
                                          layers[["sigma_2"]][[numLayers]],
                                          layers[["x"]][[numLayers]],
                                          layers[["x_hat"]][[numLayers]],
                                          layers[["y"]][[numLayers]],
                                          layers[["gamma"]][[numLayers]],
                                          layers[["beta"]][[numLayers]])

  layers[["delta_x"]][[numLayers]] <- ret[[1]]
  layers[["delta_gamma"]][[numLayers]] <- ret[[2]]
  layers[["delta_beta"]][[numLayers]] <- ret[[3]]

  if (numLayers > 1) {
    layers[["delta_weight"]][[numLayers]] <- t(layers[["output"]][[numLayers - 1]]) %*%
      layers[["delta_y"]][[numLayers]]
  } else {
    layers[["delta_weight"]][[numLayers]] <- t(trainData) %*%
      layers[["delta_y"]][[numLayers]]
  }

  # 3. Backpropagate the error
  for(i in (numLayers-1):1){
    error <-  layers[["delta_x"]][[i+1]] %*% t(layers[["weight"]][[i + 1]])
    # zero derivatives makes sure that dropout nodes' delta functions are zeros
    layers[["delta_y"]][[i]] <- error * layers[["derivative"]][[i]]

    ret <- batch_normalization_differential(layers[["delta_y"]][[i]],
                                            layers[["mu"]][[i]],
                                            layers[["sigma_2"]][[i]],
                                            layers[["x_hat"]][[i]],
                                            layers[["y"]][[i]],
                                            layers[["gamma"]][[i]],
                                            layers[["beta"]][[i]])

    layers[["delta_x"]][[i]] <- ret[[1]]
    layers[["delta_gamma"]][[i]] <- ret[[2]]
    layers[["delta_beta"]][[i]] <- ret[[3]]

    if (i > 1) {
      layers[["delta_weight"]][[i]] <- t(layers[["output"]][[i - 1]]) %*% layers[["delta_y"]][[i]]
    } else {
      layers[["delta_weight"]][[i]] <- t(trainData) %*% layers[["delta_y"]][[i]]
    }

  }

  ret <- list()
  ret[[1]] <- layers[["delta_weight"]]
  ret[[2]] <- layers[["delta_beta"]]
  ret[[3]] <- layers[["delta_gamma"]]
  return(ret)
}
