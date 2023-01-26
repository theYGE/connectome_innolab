# create_random_covariates
set.seed(123)
library(data.table)

data <- data.table(subject_id = numeric(), age = numeric(), sex = factor())
n.obs <- 400000

dir.project <- rstudioapi::getSourceEditorContext()$path  #.../attic/create_random_corr.R
dir.project <- dirname(dir.project)
dir.project <- dirname(dir.project)
dir.data <- paste0(dir.project, .Platform$file.sep, "data")
dir.tosave <- paste0(dir.data, .Platform$file.sep, "covariates")
filename <- "covariates.csv"

if(!file.exists(dir.tosave)) {
  dir.create(dir.tosave)
}

for (subject_id in seq(n.obs)) {
   sex <- as.factor(sample(c("female", "male"),1))
   age <- sample(seq(from = 18, to = 80), 1)
   id = paste0("subject_", subject_id, collapse = "")
   data <- rbindlist(list(
     data,
     data.table(fc_id = id, age = age, sex = sex)
   ))
}

fwrite(data, file = paste0(dir.tosave, .Platform$file.sep, filename))