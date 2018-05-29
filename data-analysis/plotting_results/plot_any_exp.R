#############################################
###      General Plotting Function
###
### 1) Loading more specific plotting functions
### 2) Loading analysis_helper_functions.py: loading raw data
### 3) Define main plotting function
###   - setting up data loading 
###   - load data
###   - preparing plotting parameter selection
###   - load plotting_parameter.py for parameter setting
###   - call more specific plotting function
###      -> these will use the parameters defined in plotting_parameters.py
############################################

# import more specific plotting functions
source('plot_accuracy.R')
source("plot_homing.R")
source("plot_additional_metric.R")
source("analysis_helper_functions.R")

plot_any_exp = function (exp_name, metric='accuracy', use_sums=TRUE,
                         plot_legend=FALSE, main=NULL, coherence='10', finetuning=FALSE) {
  # exp_name: name of the experiment to be plotted
  # metric: metric to be plotted (accuracy and response_distribution set additional to FALSE)
  # additional: plot one of the additional metrics, only including DNNs? 
  # use_sums: use summation over softmax or highest softmax? (ignored if additional=TRUE)
  # main: plot title
  # coherence: coherence parameter, if plotting eidolon experiment
  # finetuning: whether the raw data should be taken from the finetuning results folder
  
  # make function parameters global, so that they can be used by other functions
  # same for other variables defined below
  exp_name <<- exp_name
  metric <<- metric
  additional <<- !(metric %in% c('accuracy', 'homing'))
  use_sums <<- (use_sums && !finetuning)
  plot_legend <<- plot_legend
  main <<- main
  coherence <<- coherence
  finetuning <<- finetuning
  
  # check for wrong parameter input
  if (!(exp_name %in% exp_names)) {
    stop('Please choose an experiment from the exp_names list.')
  }
  if (!(metric %in% metrics)) {
    stop('Please choose a metric from the metrics list.')
  }
  if (additional) {
    use_sums <- FALSE
  }
  if (metric == 'entropy_1000') {use_sums<-FALSE}
  if (finetuning) {
    use_sums <- FALSE
  }
  
  # adjust DATAPATH
  if (finetuning) {
    DATAPATH <<- paste(datapath, 'fine-tuning/', sep = '')
    dnn_subjects <- c("sixteen01v4", "all-noise", "specialised")
    acc_path <<- paste(accuracies_path, "fine-tuning/", sep="")
  } else {
    DATAPATH <<- paste(datapath, 'TF/', sep = '')
    dnn_subjects <- c("googlenet", "vgg19", "resnet152")
    acc_path <<- paste(accuracies_path, "TF/", sep="")
  }
  
  # additional metrics can only be plotted for DNNs
  if (additional || (exp_name == 'salt-and-pepper-png')) {
    subjects <<- dnn_subjects
    folder_path <<- paste(figurespath, exp_name, "/", sep="")
    #folder_path <<- paste(figurespath, "additional/", exp_name, "/", sep="")
  } else {
    subjects <<- c("humans", dnn_subjects)
    folder_path <<- paste(figurespath, exp_name, "/", sep="")
  }
  
  # create output folder, if it doesn't exist already
  #if (!dir.exists(paste(figurespath, "additional/", sep=""))) { 
  #  dir.create(paste(figurespath, "additional/", sep=""))
  #}
  if (!dir.exists(folder_path)) { 
    dir.create(folder_path)
  }
  
  ################################################
  ###    loading & preprocessing experimental data
  ################################################
  
  # load data for humans and DNNs separately
  if (!additional && (exp_name != 'salt-and-pepper-png')) {
    # there is no human data for colour-png and noise-png
    if (finetuning && (exp_name %in% c('colour-png', 'noise-png'))) {
      exp_name_ = unlist(strsplit(exp_name, split='-', fixed=TRUE))[1]
      human_data <- get.expt.data(paste(exp_name_, "-experiment", sep = ""), onlyHumans=TRUE)
    } else {
      human_data <- get.expt.data(paste(exp_name, "-experiment", sep = ""), onlyHumans=TRUE)
    }
  }
  dnn_data <- get.expt.data(paste(exp_name, "-experiment", sep = ""), onlyDNNs=TRUE)
  
  # select data for plotting
  if (exp_name == 'salt-and-pepper-png') {
    expdata <<- dnn_data
  } else if (additional) {
    expdata <<- dnn_data
    # defining new column for metric of interest
    expdata$metric <<- expdata[[metric]]
  } else {
    # use highest probabilities or sums over probabilities?
    if (use_sums) {
      dnn_data$object_response <- dnn_data$object_response_sums
      dnn_data$is.correct <- dnn_data$is.correct_sums
    }
    
    # reduce dnn data to human columns and concatenate dataframes
    dnn_data <- dnn_data[colnames(human_data)]
    expdata <<- rbind(human_data, dnn_data[colnames(human_data)])
  }
  
  ############################################
  ###      Set Plotting Parameters
  ############################################
  
  # Load hard-coded plotting parameter
  source("plotting_parameters.R")
  
  ############################################
  ###       Plot Metric
  ############################################
  
  if (metric == 'accuracy') {
    plot_accuracy()
  } else if (metric == 'homing') {
    plot_homing()
  } else {
    plot_additional_metric()
  }
  
}
