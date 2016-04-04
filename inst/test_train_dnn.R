input <- matrix(runif(100), 50, 2)
input_valid <- matrix(runif(10), 5, 2)
target <- rowSums(input + input^2)
target_valid <- rowSums(input_valid + input_valid^2)

darch <- new_dnn(c(2,5,1))
darch <- train_dnn(darch,
          input,
          target,
          # input_valid,
          # target_valid,

          learn_rate_weight = exp(-5),
          learn_rate_bias = exp(-5),
          learn_rate_gamma = exp(-5),
          batch_size = 10,
          batch_normalization = F,
          dropout_input = 0,
          dropout_hidden = 0,
          momentunm_initial = 0.6,
          momentum_final = 0.9,
          momentum_switch = 100,
          num_epochs = 100,

          # target types
          error_function = meanSquareErr,
          report_classification_error = F
)

# test the dropout



# Test case 1:
# first train with BN on
# then train with BN off
rm(darch)
darch <- new_dnn(c(2, 5, 10, 1))
darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = T,
                   dropout_input = 0.4,
                   dropout_hidden = 0.8,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)

darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = F,
                   dropout_input = 0.4,
                   dropout_hidden = 0.8,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)

plot(darch@stats$dataErrors$raw)
plot(darch@stats$validErrors$raw)

# Test case 2:
# firt train with BN off
# then trian with BN on

rm(darch)
darch <- new_dnn(c(2, 5, 10, 1))
darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = F,
                   dropout_input = 0.4,
                   dropout_hidden = 0.8,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)


darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = T,
                   dropout_input = 0.4,
                   dropout_hidden = 0.8,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)

plot(darch@stats$dataErrors$raw)
plot(darch@stats$validErrors$raw)

# Test case 3:
# Compare BN training with no BN training

input <- matrix(runif(500), 250, 2)
input_valid <- matrix(runif(50), 25, 2)
target <- rowSums(cos(input) + sin(input)^2)
target_valid <- rowSums(cos(input_valid) + sin(input_valid)^2)


rm(darch)
darch <- new_dnn(c(2, 10, 10, 1))
darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = T,
                   dropout_input = 0.,
                   dropout_hidden = 0.,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)


rm(darch)
darch <- new_dnn(c(2, 10, 10, 1))
darch <- train_dnn(darch,
                   input,
                   target,
                   input_valid,
                   target_valid,
                   # training parameters
                   learn_rate_weight = exp(-5),
                   learn_rate_bias = exp(-5),
                   learn_rate_gamma = exp(-5),
                   batch_size = 10,
                   batch_normalization = F,
                   dropout_input = 0.,
                   dropout_hidden = 0.,
                   momentunm_initial = 0.6,
                   momentum_final = 0.9,
                   momentum_switch = 100,
                   num_epochs = 50,
                   # target types
                   error_function = meanSquareErr,
                   report_classification_error = F
)


# Test case 4:
# Compare train_dnn with darch
# differences:
# 1) batch generation
# 2) batch normalization
# 3) Cross Entropy Error

