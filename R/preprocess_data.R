#' Data proprosess function that covnerts a categorical input to continuous input or
#' vectorize it
#'
#' Proprosess a data set. It converts categorical data into binary variables
#' if it is unordered or continuous variable from 0 to 1 if it is ordinal
#' @param x input variable
#' @param type ordinal or other
#' @param ordered_list the rank ordering of an ordinal variable. Users are expected to
#' provide a complete list of the rank ordering. Otherwise, a default rank ordering
#' will be used.
#' @param var_name the name of the input variable. This is used to to create vectorized
#' input variables
#' @param ... other inputs
#'
#' @export

convert_categorical <- function(x,
                                type = "ordinal",
                                ordered_list = list(),
                                var_name = "var",
                                ...) {

  if(type == "ordinal") {
    unique_x <- unique(x)

    if(is.null(ordered_list)) {
      ordered_list <- sort(unique_x) # list_x has all unique values in vector x
    }

    if(any(!(unique_x %in% ordered_list))) {
      ordered_list <- sort(unique_x) # list_x has all unique values in vector x
    }

    num_categories <- length(ordered_list)
    mapped_value <- c(0:(num_categories - 1))/(num_categories - 1)
    ret <- mapped_value[match(x, ordered_list)]
  } else {
    unique_x <- unique(x)
    ordered_list <- sort(unique_x)
    mapped_value <- c(1:num_categories)
    numeric_x <- mapped_value[match(x, ordered_list)]
    vectorized_x <- matrix(0, nrow = length(x), ncol = length(unique_x))
    for( i in 1:length(x)) {
       vectorized_x[i, numeric_x[i]] <- 1
    }

    ret <- data.frame(vectorized_x)
    colnames(ret) <- paste0(var_name, " = ", ordered_list)
  }

  return(ret)
}





