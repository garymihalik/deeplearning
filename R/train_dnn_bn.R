#' Training function with batch normalization on
#'
#' Train a deep neural network with batch normalization
#'
#'
#' @export

train_dnn_bn <- function(darch,
                         input,
                         target,
                         num_epochs,
                         learn_rate_gamma = exp(-10),
                         error_function = meanSquareErr) {
  dataset <- createDataSet(input, target)

  darch@fineTuneFunction = finetune_SGD_bn

  darch = fineTuneDArch(darch, dataset,
                         dataSetValid = NULL,
                         numEpochs = num_epochs,
                         bootstrap = F,
                         isBin = F,
                         isClass = F,
                         stopErr = -Inf,
                         stopClassErr = -Inf,
                         stopValidErr = -Inf,
                         stopValidClassErr = 101,
                        errorFunc = error_function # parameter for finetune_SGD_bn
                        )
  # Calcualte the mu and sigma of each layer using the entire input data

  numLayers <- length(getLayers(darch))
  numRows <- dim(input)[[1]]

  data <- input

  for(i in 1:numLayers) {
    ret <- getLayerWeights(darch, i)
    dimV_input <- dim(ret)[[1]] - 1
    dimV_output <- dim(ret)[[2]]
    if(i < length(layers)){
      weight <- ret[1:(dimV_input), ] * (1 - darch@dropoutHidden)
      beta <- ret[(dimV_input + 1), ] * (1 - darch@dropoutHidden)
      gamma <- getLayer(darch, i)[[4]] * (1 - darch@dropoutHidden)
    } else {
      weight <- ret[1:(dimV_input), ]
      beta <- ret[(dimV_input + 1), ]
      gamma <- getLayer(darch, i)[[4]]
    }

    beta <- verticalize(beta, numRows)
    gamma <- verticalize(gamma, numRows)

    x <- data %*% weight
    ret <- batch_normalization(x, gamma, beta)
    mu <- ret[[1]]
    sigma_2 <- ret[[2]]
    y <- ret[[4]]
    unit_matrix <- diag(dim(y)[[2]])
    ret <- getLayer(darch, i)[[2]](y, unit_matrix)
    data <- ret[[1]]

    darch@layers[[i]][[5]] <- mu[1,]
    darch@layers[[i]][[6]] <- sigma_2[1,]
  }

  return (darch)
}
