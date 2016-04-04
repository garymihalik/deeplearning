#' Creates a matrix by repeating a row vector N times
#'
#' helper function that repeat a row vector N times
#'
#' @param vector the row vector
#' @param N number of rows in the output matirx
#' @return a matrix
#' @export
#'
verticalize <- function(vector, N) {
  return(matrix(rep(vector, N), N, byrow = T))
}
