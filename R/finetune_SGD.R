#' Finetune function that trains a deep neural network using stochastic gradient descent method
#'
#' This function finetunes a DArch network using SGD approach
#'
#' @export

finetune_SGD <- function(darch, trainData, targetData, errorFunc = meanSquareErr) {
  ret <- backpropagate_delta(darch, trainData, targetData, errorFunc)
  outputs <- ret[[1]]
  delta <- ret[[2]]
  layers <- getLayers(darch)
  numLayers <- length(layers)
  stats <- getStats(darch)

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
    weightsChange <- weightsInc + (getMomentum(darch) *
                                     layers[[i]][[3]][] * getDropoutMask(darch, i-1)
                                   )

    weights <- weights - weightsChange
    biasesInc <- learnRateBiases * (rowSums(t(delta[[i]])))
    biases <- biases - biasesInc
    setLayerWeights(darch,i) <- rbind(weights,biases)
    setLayerField(darch,i,3) <- weightsInc
  }

  setStats(darch) <- stats
  return(darch)

}
