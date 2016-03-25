#' function that calculates the delta function of a darch object
#'
#' Calculates the delta functions using backpropagation
#'
#' @export

backpropagate_delta <- function(darch, trainData,targetData, errorFunc = meanSquareErr) {
  layers <- getLayers(darch)
  numLayers <- length(layers)
  delta <- list()
  outputs <- list()
  derivatives <- list()
  stats <- getStats(darch)

  # apply input dropout mask to data
  # TODO same input dropout mask for all data in a batch?
  trainData <- applyDropoutMask(trainData, getDropoutMask(darch, 0))

  # 1. Forwardpropagate
  data <- trainData
  numRows <- dim(data)[1]
  for (i in 1:numLayers){
    data <- cbind(data,rep(1,numRows))
    func <- getLayerFunction(darch, i)
    weights <- getLayerWeights(darch, i)

    if (i < numLayers)
    {
      weights <- applyDropoutMask(weights, getDropoutMask(darch, i))
      # RZ: is this correct?
    }

    ret <- func(data,weights)

    # apply dropout masks to output, unless we're on the last layer
    if (i < numLayers)
    {
      ret[[1]] <- applyDropoutMask(ret[[1]], getDropoutMask(darch, i))
      ret[[2]] <- applyDropoutMask(ret[[2]], getDropoutMask(darch, i))
    }

    outputs[[i]] <- ret[[1]]
    data <- ret[[1]]
    derivatives[[i]] <- ret[[2]]
  }
  rm(data,numRows)

  # 2. Calculate the Error on the network output
  errorDerivative <- (outputs[[numLayers]] - targetData)
  delta[[numLayers]] <- errorDerivative * derivatives[[numLayers]]

  E <- errorFunc(targetData,outputs[[numLayers]][])[[1]]
  #flog.debug(paste("Error",E[[1]],E[[2]]))

  # 4. Backpropagate the error
  for(i in (numLayers-1):1){
    weights <- layers[[i+1]][[1]][]
    # remove bias row
    weights <- weights[1:(nrow(weights)-1),,drop=F]

    error <-  matMult(delta[[i+1]], t(weights))
    delta[[i]] <- error * derivatives[[i]]
  }

  ret <- list()
  ret[[1]] <- outputs
  ret[[2]] <- delta
  return(ret)
}
