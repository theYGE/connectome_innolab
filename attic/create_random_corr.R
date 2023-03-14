# create random (positive) correlation matrix
set.seed(123)
library(randcorr)
library(corpcor)

n.roi <- 400  # number of ROIs
zeros <- c(0.1, 0.15)  # sample range for percentages of zeros
zeros.granularity <- 0.001  # defines interval points in zeros
n.obs <- 1000
set.zeros <- FALSE
dir.project <- rstudioapi::getSourceEditorContext()$path  #.../attic/create_random_corr.R
dir.project <- dirname(dir.project)
dir.project <- dirname(dir.project)
dir.data <- paste0(dir.project, .Platform$file.sep, "data")
dir.corr.tosave <- paste0(dir.data, .Platform$file.sep, "corr_mat")
dir.parcorr.tosave <- paste0(dir.data, .Platform$file.sep, "parcorr_mat")

if(!file.exists(dir.corr.tosave)) {
  dir.create(dir.corr.tosave)
}
if(!file.exists(dir.parcorr.tosave)) {
  dir.create(dir.parcorr.tosave)
}

for (i in seq(n.obs)) {
  if (set.zeros) {
    zeros.percentage <- sample(seq(from = zeros[1], zeros[2], zeros.granularity), 1)
    zeros.num <- ceiling(zeros.percentage * n.roi) 
    zeros.coord1 <- sample(seq(from = 1, to = n.roi), zeros.num)
    zeros.coord2 <- sample(seq(from = 1, to = n.roi), zeros.num)
    
    while (any(zeros.coord2 == zeros.coord1)) {
      cat(paste0("Have to resample to avoid 0 on diagonal in iteration ", i, "\n"))
      zeros.coord1 <- sample(seq(from = 1, to = n.roi), zeros.num)
    }
  }
  filename.corr <- paste0(dir.corr.tosave, .Platform$file.sep, "subject_", i, ".csv")
  filename.parcorr <- paste0(dir.parcorr.tosave, .Platform$file.sep, "rand_pcorr_", i, ".csv")
  
  corrmat <- randcorr(n.roi)  # only positive
  # corrmat[zeros.coord1, zeros.coord2] <- 0
  parcorrmat <- cor2pcor(corrmat)
  write.table(corrmat, file = filename.corr, col.names = TRUE, row.names = FALSE, sep = ",")
  #write.table(parcorrmat, file = filename.parcorr, col.names = TRUE, row.names = FALSE, sep = ",")
  print(paste0("finished sample number", i, collapse = ""))
}