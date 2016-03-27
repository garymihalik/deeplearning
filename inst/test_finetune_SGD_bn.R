input <- matrix(runif(250), 50, 5)
target <- rowSums(cos(input) + sin(input)^2)

mean_v <- mean(target)
target <- as.numeric(target > mean_v * 1.02 )


darch = darch( x = input,
                 y = target,
                 layers = c(5, 100, 50, 1),
                 # darch = darch,
                 darch.layerFunctionDefault = rectified_linear_unit_function,
                 darch.layerFunctions = c("3" = linearUnitDerivative),
                 darch.bootstrap = F,
                 darch.isBin = F,
                 darch.isClass = F,
                 darch.learnRateWeights = 0.01,
                 darch.learnRateBiases = 0.01,
                 darch.dropoutInput = 0.,
                 darch.dropoutHidden = 0.,
                 darch.fineTuneFunction = backpropagation, # finetune_SGD,
                 darch.batchSize = 10,
                 darch.numEpochs = 1
)

backpropagate_delta_bn(darch,
                       input,
                       target)

darch <- finetune_SGD_bn(darch,
                input,
                target)

darch@executeFunction <- run_darch_bn

darch <- train_dnn_bn(darch,
                      input,
                      target,
                      num_epochs = 50)



