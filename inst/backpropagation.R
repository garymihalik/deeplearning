test_backpropagation <- function(darch, trainData, targetData, ...)
{
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

    # apply dropout masks to weights, unless we're on the last layer; this is
    # done to allow activation functions to avoid considering values that are
    # later going to be dropped
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
  # TODO if we use dropout in the output layer, multiply dropout mask in here
  error <- (targetData - outputs[[numLayers]][])
  delta[[numLayers]] <- error * derivatives[[numLayers]]

  E <- getErrorFunction(darch)(targetData,outputs[[numLayers]][])
  #flog.debug(paste("Error",E[[1]],E[[2]]))

  # 4. Backpropagate the error
  for(i in (numLayers-1):1){
	  weights <- layers[[i+1]][[1]][]
    # remove bias row
	  weights <- weights[1:(nrow(weights)-1),,drop=F]

	  error <-  (delta[[i+1]] %*% t(weights))
	  delta[[i]] <- error * derivatives[[i]]
  }

  # 5.  Update the weights
  learnRateBiases <- getLearnRateBiases(darch)
  learnRateWeights <- getLearnRateWeights(darch)
  for(i in numLayers:1){
    weights <- layers[[i]][[1]][]
    biases <- weights[nrow(weights),,drop=F]
    weights <- weights[1:(nrow(weights)-1),,drop=F]

    # Check if the weightInc field in the layer list exists.
    if (length(layers[[i]]) < 3){
      layers[[i]][[3]] <- matrix(0,nrow(weights),ncol(weights))
    }

    if (i > 1){
      output <- outputs[[i-1]]
    }else{
      output <- trainData
    }

    weightsInc <- t(learnRateWeights * (t(delta[[i]]) %*% output))

    # apply dropout mask to momentum
    weightsChange <- weightsInc + (getMomentum(darch) * layers[[i]][[3]][]
      * getDropoutMask(darch, i-1))

    weights <- weights + weightsChange
    biasesInc <- learnRateBiases * (rowSums(t(delta[[i]])))
    biases <- biases + biasesInc
    setLayerWeights(darch,i) <- rbind(weights,biases)
    setLayerField(darch,i,3) <- weightsInc
  }

  setStats(darch) <- stats
  return(darch)
}
