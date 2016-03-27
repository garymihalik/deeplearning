#' Finetune function that trains a deep neural network using stochastic gradient descent method
#'
#' This function finetunes a DArch network using SGD approach
#'
#' @export

finetune_SGD_bn <- function(darch,
                            trainData,
                            targetData,
                            learn_rate_gamma = exp(-10),
                            errorFunc = meanSquareErr) {
    # stats <- getStats(darch)

    ret <- backpropagate_delta_bn(darch, trainData, targetData, errorFunc)
    delta_weight <- ret[[1]]
    delta_beta <- ret[[2]]
    delta_gamma <- ret[[3]]

    learnRateBiases <- getLearnRateBiases(darch)
    learnRateWeights <- getLearnRateWeights(darch)
    learnRateGamma <- learn_rate_gamma

    numLayers <- length(delta_weight)

    for(i in numLayers:1) {
      weights <- getLayer(darch, i)[[1]]
      biases <- weights[nrow(weights),,drop=F]
      weights <- weights[1:(nrow(weights)-1),,drop=F]

      if(length(getLayer(darch, i)) < 4) {
        gamma <- matrix(1, nrow = 1, ncol = dim(weights)[[2]])
      } else {
        gamma <- getLayer(darch, i)[[4]]
      }

      if (length(getLayer(darch, i)) < 3) {
        weightsChange_prev <- matrix(0,nrow(weights),ncol(weights))
      } else {
        weightsChange_prev <- getLayer(darch, i)[[3]]
      }

    # Calculate the change in weights
    # apply dropout mask to momentum
    weightsInc <- (learnRateWeights * delta_weight[[i]])
    weightsChange <- weightsInc + (getMomentum(darch) *
                            weightsChange_prev * getDropoutMask(darch, i-1)
    )
    weights <- weights - weightsChange

    # Calculate the change in beta (biases)
    biasesInc <- learnRateBiases * delta_beta[[i]][1,]
    biases <- biases - biasesInc

    # Calculate the change in gamma
    gammaInc <- learnRateGamma * delta_gamma[[i]][1,]
    gamma <- gamma - gammaInc

    setLayerWeights(darch,i) <- rbind(weights,biases)
    setLayerField(darch,i,3) <- weightsInc
    setLayerField(darch, i, 4) <- gamma
  }

  # setStats(darch) <- stats
  return(darch)

}
