#' Finetune function that trains a deep neural network using stochastic gradient descent method
#'
#' This function finetunes a DArch network using SGD approach
#'
#' @export

finetune_SGD_bn <- function(darch, trainData, targetData, errorFunc = meanSquareErr) {
    ret <- backpropagate_delta_bn(darch, trainData, targetData, errorFunc)
    delta_weight <- ret[[1]]
    delta_beta <- ret[[2]]

    learnRateBiases <- getLearnRateBiases(darch)
    learnRateWeights <- getLearnRateWeights(darch)
    numLayers <- length(delta_weight)

    for(i in numLayers:1) {
      weights <- getLayer(darch, i)[[1]]
      biases <- weights[nrow(weights),,drop=F]
      weights <- weights[1:(nrow(weights)-1),,drop=F]

      if (length(getLayer(darch, i)) < 3) {
        weightsChange_prev <- matrix(0,nrow(weights),ncol(weights))
      } else {
        weightsChange_prev <- getLayer(darch, i)[[3]]
      }

    # apply dropout mask to momentum
    weightsInc <- (learnRateWeights * delta_weight[[i]])
    weightsChange <- weightsInc + (getMomentum(darch) *
                            weightsChange_prev * getDropoutMask(darch, i-1)
    )
    weights <- weights - weightsChange

    biasesInc <- learnRateBiases * delta_beta[[i]]
    biases <- biases - biasesInc

    setLayerWeights(darch,i) <- rbind(weights,biases)
    setLayerField(darch,i,3) <- weightsInc
  }

  setStats(darch) <- stats
  return(darch)

}
