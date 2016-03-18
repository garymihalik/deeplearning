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
#' @param input Input data
#'
#' @export

forward_propagate <- function(darch, input = darch@dataSet@data) {
  numLayers <- length(getLayers(darch))
  h <- list()
  derivative <- list()
  h[[1]] <- input
  derivative[[1]] <- matrix(0, dim(input)[[1]], dim(input)[[2]])
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

#' Utility function that prints out the weight of a deep neural network
#'
#' Prints out the weight in a heat map, 3D surface, or histogram
#'
#' @param darch DArch instance
#' @param num_of_layer the number of the layer to print
#' @param show_derivative T to show the weight value. F to show the percentage
#' weight change in the finetuning stage. This helps spot the network saturation problem.
#' @param type type of the graph. It supports "heatmap", "surface", and "histogram"
#'
#' @export

print_weight <- function(darch, num_of_layer, show_derivative = F, type = "heatmap") {
  weight <- getLayer(darch, num_of_layer)[[1]]
  weight_change <- getLayer(darch, num_of_layer)[[3]] / weight[1:(dim(weight)[[1]]-1), ]

  if(type == "histogram") {
    if(!show_derivative) {
      plotly::plot_ly(x = c(weight), type = type)
    } else {
      plotly::plot_ly(x = c(weight_change), type = type)
    }
  } else {
    if(!show_derivative) {
      plotly::plot_ly(z = weight, type = type, colorscale = "hot")
    } else {
      plotly::plot_ly(z = weight_change, type = type, colorscale = "hot")
    }
  }
}

#' Utility function that prints out the layer value for a neural net
#'
#' Prints out the network's layer activation value
#'
#' @param darch a DArch instance
#' @param input an input dataset. Defaults to darch's input data
#' @param num_of_layer the number of the layer to print
#' @param show_derivative whether to show the activation or the derivative
#' @param type type of the graphs to print. It can be heatmap, surface, or histogram
#'
#' @export

print_layer <- function(darch, num_of_layer, input = darch@dataSet@data,
                        show_derivative = F,
                        type = "heatmap") {
  fprop <- forward_propagate(darch, input)
  layer_activation <- fprop[[1]][[num_of_layer + 1]]
  layer_derivative <- fprop[[2]][[num_of_layer + 1]]

  if(type == "histogram") {
    if(!show_derivative) {
      plotly::plot_ly(x = c(layer_activation), type = type)
    } else {
      plotly::plot_ly(x = c(layer_derivative), type = type)
    }
  } else {
    if(!show_derivative) {
      plotly::plot_ly(z = layer_activation, type = type, colorscale = "hot")
    } else {
      plotly::plot_ly(z = layer_derivative, type = type, colorscale = "hot")
    }
  }
}

#' Utility function that calculates the Accuracy Ratio of a binary classification
#'  model
#'
#' Calculates the Accruacy Ratio
#'
#' @param x model
#'
#' @export

AR <- function(x, ...) {
  UseMethod("AR")
}

#' Utility function that calculates the Accuracy Ratio of a binary classification
#'  model
#'
#' Calculates the Accruacy Ratio of a given set of probability
#'
#' @param x a list of probability
#' @param target target variable
#'
#' @export

AR.numeric <- function(x, target) {
  AR.default(x, target)
}

#' Utility function that calculates the Accuracy Ratio of a binary classification
#'  model
#'
#' Calculates the Accruacy Ratio of a given set of probability
#'
#' @param x a list of probability
#' @param target target variable
#'
#' @export

AR.default <- function(x, target) {
  N <- length(x)
  seq = order(x, decreasing = T)
  target <- target[seq]
  auc <- 0
  totTarget <- sum(target)
  y <- c()
  for (i in 1:N) {
    lorenzeCurve <- sum(target[1:i]) / totTarget
    auc <- auc + lorenzeCurve * 1 / N
    y <- cbind(y, lorenzeCurve)
  }
  auc <- auc
  pd <- sum(target) / N
  ar <- (2 * auc - 1) / (1 - pd)
  plot(as.vector(y))
  return (ar)
}


#' Utility function that calculates the Accuracy Ratio of a binary classification
#'  model
#'
#' Calculates the Accruacy Ratio of a given set of probability
#'
#' @param x an instance of DArch class
#' @param input input variable
#' @param target target variable
#'
#' @export


AR.DArch <- function(darch, input = darch@dataSet@data,
                     target = darch@dataSet@targets) {
  pred <- predict(darch, newdata = input)
  AR.default(pred, target)
}















