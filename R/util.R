# Calculate the RSQ of a regression model


#' Utilitiy function that calcualtes RSQ
#'
#' Calcualte a regression model's RSQ
#'
#' @param  x Regression Model
#' @param  ... Additional Input
#'
#' @import  futile.logger
#' @export

rsq <- function(x, ...) {
  UseMethod("rsq", x)
}

#' Utilitiy function that calcualtes RSQ of a DArch instance
#'
#' Calcualte a regression model's RSQ
#'
#' @param  x DArch Model
#' @param  input Input data
#' @param  target Target data
#'
#' @import  futile.logger
#' @export

rsq.DArch <- function(darch,
                      input = darch@dataSet@data,
                      target = darch@dataSet@targets) {
  y <- target
  pred <- predict(darch, newdata = input)
  plot(y, pred)
  RSQ <- 1 - sum((pred-y)^2)/sum((y-mean(y))^2)
  flog.info(paste0("RSQ = ", RSQ))
}

#' Utilitiy function that calcualtes RSQ of a linear model
#'
#' Calcualte a regression model's RSQ
#'
#' @param  x linear Model
#' @param  input Input data
#' @param  target Target data
#'
#' @import  futile.logger
#' @export

rsq.lm <- function(mod, input, target) {
  y <- target
  pred <- predict(mod, newdata = data.frame(input))
  plot(y, pred)
  RSQ <- 1 - sum((pred-y)^2)/sum((y-mean(y))^2)
  flog.info(paste0("RSQ = ", RSQ))
}


#' Forward propagate a deep neural network
#'
#' Calcualte all layer's activation value and derivative function value
#'
#' @param darch A DArch instance
#' @param data Input data
#'
#' @export

forward_propagate <- function(darch, data) {
  numLayers <- length(getLayers(darch))
  h <- list()
  derivative <- list()
  h[[1]] <- data
  derivative[[1]] <- matrix(0, dim(data)[[1]], dim(data)[[2]])
  for(i in 1:(numLayers)) {
    ret <- getLayer(darch, i)
    weight <- ret[[1]]
    layerFunc <- ret[[2]]
    h[[i]] <- cbind(h[[i]], rep(1, dim(x)[[1]]))
    o <- layerFunc(h[[i]], weight)
    h[[i]] <- h[[i]][, 1:(dim(h[[i]])[[2]]-1)]
    h[[i+1]] <- o[[1]]
    derivative[[i+1]] <- o[[2]]
  }
  ret <- list()
  ret[[1]] <- h
  ret[[2]] <- derivative
  return(ret)
}

