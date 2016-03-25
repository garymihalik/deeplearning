# New a DArch instance
source("inst/dropout.R")


darch = newDArch(c(10,20,1), batchSize = 10)
setDropoutOneMaskPerEpoch(darch) = F
setFineTuneFunction(darch) <- minimizeClassifier
setFineTuneFunction(darch) <- backpropagation
setFineTuneFunction(darch) <- backpropSGD
darch = generateDropoutMasksForDarch(darch)

# New a dataset
input <- matrix(runif(200), 10, 20)
target <- rowSums(cos(input) + sin(input))

dataset = createDataSet(input, target)

# Use the interface function

darch = darch( x = dataset@data,
               y = dataset@targets,
               layers = c(20, 100, 50, 1),
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
               darch.numEpochs = 10
               )

plot(predict(darch), target)

# Finetune the DArch instance
darch3 = fineTuneDArch(darch, dataset,
                       dataSetValid = NULL,
                       numEpochs = 5,
                       bootstrap = F,
                       isBin = T,
                       isClass = T,
                       stopErr = -Inf,
                       stopClassErr = -Inf,
                       stopValidErr = -Inf,
                       stopValidClassErr = 101
                       )

# Use the fineTune function directly

darch2 = darch

# Backpropagation/ Steepest Descent
darch2 = backpropagation(darch, dataset@data, dataset@targets)

# Conjugate Gradient Descent - Doesn't seem to work well. A bug in the code?
# darch3 = backpropCGD(darch, dataset@data, dataset@targets, length = 3, switchLayers = 0)

# Modified Steepest Gradient Descent
darch2 <- backpropSGD(darch, dataset@data, dataset@targets, crossEntropyErr)

testFunc2(darch2, dataset@data, dataset@targets, "Train set")

getLayer(darch,1)[[1]][1,]
getLayer(darch2,1)[[1]][1,]
getLayer(darch3,1)[[1]][1,]

testFunc2(darch3, dataset@data, dataset@targets, "Train Set")


gr1 <- calcGradient(par, darch2, dims, data, target, crossEntropyErr, epochSwitch)
gr2 <- fr(par, darch2, dims, data, target, epochSwitch )
gr1 - gr2

