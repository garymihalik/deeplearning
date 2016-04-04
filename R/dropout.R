#' Generates dropout masks for dnn
#'
#' Generates dropout maks for dnn
#'
#'
#'
#' @export

generateDropoutMasksForDarch <- function(darch, dropout_input, dropout_hidden)
{
  dropoutMasks <- list()
  numLayers <- length(getLayers(darch))
  # generate dropout masks
  darch@dropoutMasks[[1]]<-
    generateDropoutMask(nrow(getLayerWeights(darch, 1)[]) - 1,
                        dropout_input)

  for (i in 1:(numLayers - 1))
  {
    setDropoutMask(darch, i) <-
      generateDropoutMask(nrow(getLayerWeights(darch, i+1)[])-1,
                          dropout_hidden)
  }

  return (darch)
}

#' Generates the dropout mask for the deep neural network
#'
#' Generates the dropout mask for the deep neural network
#'
#'
#' @export


generateDropoutMask <- function(length, dropoutRate)
{
  if (dropoutRate == 0)
  {
    ret <- rep(1, length)
  }
  else
  {
    ret <- sample(c(0, 1/(1 - dropoutRate)), length, replace = T,
                  prob = c(dropoutRate, 1 - dropoutRate))
  }

  return (ret)
}
