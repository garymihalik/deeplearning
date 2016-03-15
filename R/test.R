x <- matrix(runif(200), 40, 5)
y <- rowSums(x)

darch1 <- darch(x, y,
                layers = c(5, 100, 100, 1),
                rbm.numEpochs = 0,
                darch.bootstrap =  F,
                darch.layerFunctionDefault = rectified_linear_unit_function,
                # darch.layerFunctionDefault = sigmoidUnitDerivative,
                darch.layerFunctions = c("3" = linearUnitDerivative),
                darch.isBin = F,
                darch.isClass = F,
                darch.batchSize = 10,
                darch.numEpochs = 100
                )

plot(y, predict(darch1))
