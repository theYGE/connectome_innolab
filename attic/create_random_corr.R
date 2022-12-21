# create random (positive) correlation matrix
set.seed(123)
library(randcorr)

n.roi <- 400  # number of ROIs
zeros <- c(0.1, 0.15)  # sample range for percentages of zeros
zeros.granularity <- 0.001  # defines interval points in zeros
n.obs <- 1000
dir.project <- rstudioapi::getSourceEditorContext()$path  #.../attic/create_random_corr.R
dir.project <- dirname(dir.project)
dir.project <- dirname(dir.project)
dir.data <- paste0(dir.project, .Platform$file.sep, "data")
dir.tosave <- paste0(dir.data, .Platform$file.sep, "corr_mat")

if(!file.exists(dir.tosave)) {
  dir.create(dir.tosave)
}

for (i in seq(n.obs)) {
  zeros.percentage <- sample(seq(from = zeros[1], zeros[2], zeros.granularity), 1)
  zeros.num <- ceiling(zeros.percentage * n.roi) 
  zeros.coord1 <- sample(seq(from = 1, to = n.roi), zeros.num)
  zeros.coord2 <- sample(seq(from = 1, to = n.roi), zeros.num)
  
  while (any(zeros.coord2 == zeros.coord1)) {
    cat(paste0("Have to resample to avoid 0 on diagonal in iteration ", i, "\n"))
    zeros.coord1 <- sample(seq(from = 1, to = n.roi), zeros.num)
  }
  
  filename <- paste0(dir.tosave, .Platform$file.sep, "rand_corr_", i, ".csv")
  corrmat <- abs(randcorr(n.roi))  # only positive
  corrmat[zeros.coord1, zeros.coord2] <- 0
  write.table(corrmat, file = filename, col.names = TRUE, row.names = FALSE, sep = ",")
}