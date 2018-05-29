###################################################################
#   Data analysis helper script for object-recognition experiments
#   All important functions for data analysis are collected
#   here (to be used for plotting, analysis, and in the
#   data-analysis.R script)
#   -------------------------------------------------------
#   Author:   Robert Geirhos
#   Based on: R version 3.2.3
###################################################################

library(ggplot2)

vgg.100 = rgb(0, 105, 170, maxColorValue = 255)
human.100 = rgb(165, 30, 55, maxColorValue = 255)

###################################################################
#               READING EXPERIMENTAL DATA
###################################################################

get.expt.data = function(expt.name, datapath=DATAPATH, only.humans = FALSE) {
  # Read data and return in the correct format
  
  if(is.null(datapath)) {
    stop("you need to define the DATAPATH variable")
  }
  
  dat = NULL
  expt.path = paste(datapath, expt.name, sep="")
  files = list.files(expt.path)
  
  if(length(files) < 1) {
    warning(paste("No data for expt", expt.name, "found! Check datapath."))
  }
  
  for (i in 1:length(files)) {
    add.data = TRUE
    if(only.humans) {
      if(grepl("subject", files[i])) {
        add.data = TRUE
      } else {
        add.data = FALSE
      }
    }
    if(!endsWith(files[i], ".csv")) {
      warning("File without .csv ending found (and ignored)!")
    } else if (add.data) {
      dat = rbind(dat, read.csv(paste(expt.path, files[i], sep="/")))
    }
  }
  dat$imagename = as.character(dat$imagename)
  dat$is.correct = as.character(dat$object_response) == as.character(dat$category)
  dat$is.human = ifelse(grepl("subject", dat$subj), TRUE, FALSE)
  
  return(data.frame(experiment.name = expt.name, dat))
}

###################################################################
#               HELPER FUNCTIONS
###################################################################

endsWith = function(argument, match, ignore.case = TRUE) {
  # Return: does 'argument' end with 'match'?
  # Code adapted from:
  # http://stackoverflow.com/questions/31467732/does-r-have-function-startswith-or-endswith-like-python
  
  if(ignore.case) {
    argument = tolower(argument)
    match = tolower(match)
  }
  n = nchar(match)
  
  length = nchar(argument)
  
  return(substr(argument, pmax(1, length - n + 1), length) == match)
}

get.accuracy = function(dat) {
  # Return data.frame with x and y for condition and accuracy.
  
  tab = table(dat$is.correct, by=dat$condition)
  false.index = 1
  true.index = 2
  acc = tab[true.index, ] / (tab[false.index, ]+tab[true.index, ])
  d = as.data.frame(acc)
  
  if(length(colnames(tab)) != length(unique(dat$condition))) {
    stop("Error in get.accuracy: length mismatch.")
  }
  
  #enforce numeric ordering instead of alphabetic (otherwise problem: 100 before 20)
  if(!is.factor(dat$condition)) {
    #condition is numeric
    d$order = row.names(d)
    d$order = as.numeric(d$order)
    d = d[with(d, order(d$order)), ]
    d$order = NULL
    e = data.frame(x = as.numeric(row.names(d)), y=100*d[ , ])
  } else {
    #condition is non-numeric
    e = data.frame(x = row.names(d), y=100*d[ , ])
  }
  return(e)
}