#' Train a deep neural network
#'
#' This function trains a deep neural network
#'
#' @param darch a darch instance
#' @param input input data for training
#' @param target target data for training
#' @param input_valid input data for validation
#' @param target_valid target data for validation
#' @param ... additional input
#' @param learn_rate_weight learning rate for the weight matrices
#' @param learn_rate_bias learning rate for the biases
#' @param learn_rate_gamma learning rate for the gamma
#' @param batch_size batch size during training
#' @param batch_normalization logical value that determines whether to turn on
#'  batch normalization during training. Recommneded value: T
#' @param dropout_input dropout ratio at input layer. Recommneded value: 0.2
#' @param dropout_hidden dropout ratio at hidden layers. Recommended value: 0.5
#' @param momentum_initial momentum ratio during training. Recommended value: 0.6
#' @param momentum_final final momentum during training. Recommended value: 0.9
#' @param momentum_switch afther which epoch the final momentum ratio is used during training
#' @param num_epochs number of iterations of the training
#' @param error_function error function to minimize during training
#' @param report_classification_error logical value. T to report the classification error
#'  during training
#'
#' @return a trained deep neural network (darch instance)
#' @export


train_dnn <- function(darch, # darch instance to train
                      input, # input data matrix
                      target, # target data matrix
                      input_valid = NULL, # validation data input
                      target_valid = NULL, # validation data target
                      ...,
                      # training parameters
                      learn_rate_weight,
                      learn_rate_bias,
                      learn_rate_gamma,
                      batch_size,
                      batch_normalization,
                      dropout_input,
                      dropout_hidden,
                      momentunm_initial,
                      momentum_final,
                      momentum_switch,
                      num_epochs,

                      # target types
                      error_function,
                      report_classification_error = F
) {
  # 1. set up the inputs
  timeStart <- Sys.time()
  dataSet <- createDataSet(input, target)
  numObs <- nrow(input)

  # set the stats of darch
  if (is.null(darch@stats) || length(darch@stats) < 1){
    stats <-
      list("dataErrors"=list("raw"=c(),"class"=c()),
           "validErrors"=list("raw"=c(),"class"=c()),
           "times"= 0)

    darch@stats <- stats
  }

  trainData <- as.matrix(input)
  trainTarget <- as.matrix(target)

  if(!is.null(input_valid)) {
    validData <- as.matrix(input_valid)
    validTarget <- as.matrix(target_valid)
  } else {
    validData <- NULL
    validTarget <- NULL
  }

  if (!validateDataSet(dataSet, darch))
  {
    stop("Invalid dataset provided.")
  }

  if (!is.null(validData)) {
    if (dim(trainData)[[2]] != dim(validData)[[2]] |
        dim(as.matrix(trainTarget))[[2]] != dim(as.matrix(validTarget))[[2]]) {
      stop("Invalid validation dataset.")
    }
  }

  # 2. train the neural net
  flog.info("Start training the neural net.")
  start_epoch <- getEpochs(darch)
  flog.info(paste("The neural net has been trained ", start_epoch, " times."))

  for(epoch in (1 + start_epoch):(num_epochs + start_epoch)) {
    flog.info(paste("Epoch numebr: ", epoch))

    # make the batches
    batch <- make_batches(dim(trainData)[[1]], batch_size)
    num_batches <- max(batch[, 2])

    for(i in 1:num_batches) {
      # Generate a new dropout mask for each batch
      darch <- generateDropoutMasksForDarch(darch, dropout_input, dropout_hidden)
      # Train the neural net
      darch <- finetune_SGD_bn(darch,
                               trainData[batch[,2] == i,],
                               trainTarget[batch[,2] == i,],
                               learn_rate_weight = learn_rate_weight,
                               learn_rate_bias = learn_rate_bias,
                               learn_rate_gamma = learn_rate_gamma,
                               errorFunc = error_function,
                               with_BN = batch_normalization
                               )
    }

    # calculates the new mu and sigma of darch
    if (batch_normalization) {
      darch <- calcualte_population_mu_sigma(darch, trainData)
    } else {
      darch <- reset_population_mu_sigma(darch)
    }

    # calcualtes the error

    # training errors
    pred_train <- predict(darch, newdata = trainData)
    error_train <- error_function(pred_train, trainTarget)
    flog.info(paste(error_train[[3]], "in training:  ", error_train[[1]]))
    darch@stats$dataErrors$raw <- c(darch@stats$dataErrors$raw, error_train[[1]])

    if(report_classification_error) {
      ce_train <- classification_error(pred_train, trainTarget)
      flog.info(paste(ce_train[[2]], "in training:  ", ce_train[[1]]))
      darch@stats$dataErrors$class <- c(darch@stats$dataErrors$class, ce_train[[1]])
    }

    # validation errors
    if(!is.null(validData)) {
      pred_valid <- predict(darch, newdata = validData)
      error_valid <- error_function(pred_valid, validTarget)
      flog.info(paste(error_valid[[3]],  "in validation:", error_valid[[1]]))
      darch@stats$validErrors$raw <- c(darch@stats$validErrors$raw, error_valid[[1]])

      if(report_classification_error) {
        ce_valid <- classification_error(pred_valid, validTarget)
        flog.info(paste(ce_valid[[2]], "in validation:", ce_valid[[1]]))
        darch@stats$validErrors$class <- c(darch@stats$validErrors$class, ce_valid[[1]])
      }
    }
    # increase the epoch by 1
    darch@epochs <- darch@epochs + 1
  }
  flog.info("End of the training")

  # 3. Save the training statistics
  if (is.null(darch@stats[["times"]])) {
    darch@stats[["times"]] <- 0
  }
  darch@stats[["times"]] <- darch@stats[["times"]] + as.double(Sys.time() - timeStart, "secs")

  return (darch)
}

#' Make the batches
#'
#' This function makes the batches during training



make_batches <- function(numObs, batchsize) {
  order <- sample(1:numObs, numObs)
  group <- c()
  num_batches <- ceiling(numObs / batchsize)
  for (i in 1:numObs) {
    group <- c(group, (i %% num_batches + 1))
  }
  batch <- cbind(order, group)
  batch <- batch[order(order), ]
  return (batch)
}
