############################################
#  Main Plotting Script for Results Plots
#
#
# 1) Define experiments, metrics and datapath
# 2) Load source files:
#   - plot_any_exp: main plotting function
# 2) Call plotting function --> plot_any_exp()
#
############################################
## Define experiments, metrics and datapath
############################################

# list of experiments and subjects
exp_names <- c('false-colour', 'highpass', 'lowpass', 'phase-scrambling', 'power-equalisation', 'rotation',
               'colour', 'contrast', 'contrast-png', 'noise', 'eidolon',
               'colour-png', 'noise-png', 'salt-and-pepper-png')

metrics <- c("accuracy", "homing", "error_consistency", "confusion",
             # all below are treated as additional metrics
             # additional metrics (see below) can only be plotted for DNNs
             # prob_correct := probability of the correct class
             # rank_correct := rank of the correct class
             # entropy := entropy between class predictions
             # highest_prob := highest class probability
             # is.correct := is the prediction correct --> used for mean accuracies
             # 16 vs 1000: use 16 entry-level categories or 1000 ImageNet categories
             
             "prob_correct_16", "rank_correct_16", "entropy_16", "highest_prob_16", "is.correct",
             "prob_correct_16_sums", "rank_correct_16_sums", "entropy_16_sums", "highest_prob_16_sums", "is.correct_sums",
             "entropy_1000")

# define data path
datapath <- "../../raw-data/"
figurespath <- "../../figures/results/"
accuracies_path <- "../../raw-accuracies/"
confusionpath <- "../../figures/confusion/"

# source helper and plotting functions
source("plot_any_exp.R")

############################################
###       Uncomment to plot
###
###      Plotting accuracies
############################################

# for tensorflow
#for (exp_name in exp_names[1:10]) {
#  print(paste("Plotting TF_sums accuracy for", exp_name))
#  plot_any_exp(exp_name, metric = 'accuracy', use_sums = TRUE,
#               plot_legend=startsWith(exp_name, 'colour'))
#  #dev.off()
#}


############################################
###   Plotting accuracies after finetuning
############################################

# for tensorflow
#for (exp_name in exp_names[c(2, 3, 4, 6, 9, 13, 14, 12)]) {
#  print(paste("Plotting TF accuracy after finetuning for", exp_name))
#  plot_any_exp(exp_name, metric = 'accuracy', use_sums = FALSE,
#               plot_legend=startsWith(exp_name, 'colour'), finetuning=TRUE)
#  #dev.off()
#}


############################################
###      Plotting homing
############################################
# 
# # for tensorflow
# for (exp_name in exp_names[1:10]) {
#  print(paste("Plotting TF_sums homing for", exp_name))
#  plot_any_exp(exp_name, metric = 'homing', use_sums = TRUE,
#               plot_legend=FALSE)
#  #dev.off()
# }

############################################
###   Plotting homing after finetuning
############################################
# 
# # for tensorflow
# for (exp_name in exp_names[c(2, 3, 4, 6, 9, 12, 13, 14)]) {
#  print(paste("Plotting TF homing fter finetuning for", exp_name))
#  plot_leg=startsWith(exp_name, 'colour')
#  plot_any_exp(exp_name, metric = 'homing', use_sums = FALSE,
#               plot_legend=FALSE, finetuning=TRUE)
#  #dev.off()
# }


############################################
###      Probability vector entropy
############################################

# for tensorflow
#for (exp_name in exp_names[c(1,2,3,4,5,6,7,8,9,10)]) {
#  print(paste("Plotting TF probability vector entropy for ", exp_name))
#  plot_any_exp(exp_name, metric = 'entropy_1000', use_sums=FALSE,
#               plot_legend=startsWith(exp_name, 'colour'))
#  #dev.off()
#}

############################################
###      Eidolon plots by Coherence
############################################

# for (coherence in c('0', '3', '10')) {
#  for (metric in c('accuracy', 'homing', 'entropy_1000')) {
#    print(paste("Plotting Eidolon-experiment", metric, "for coherence", coherence))
#    plt_leg = FALSE #(metric!='homing') && ((metric!='entropy_1000') || (coherence!='3'))
#    plot_any_exp('eidolon', metric = metric, use_sums =(metric != 'entropy_1000'),
#                 plot_legend=plt_leg, coherence=coherence)
#    #dev.off()
#  }
# }



############################################
###      Error consistency plots
############################################

# for (exp_name in exp_names[1:9]) {
#   print(paste("Plotting error consistency for", exp_name))
#   plot_leg <- startsWith(exp_name, 'lowpass')
#   plot_any_exp(exp_name, metric = 'error_consistency', use_sums = TRUE,
#                plot_legend=plot_leg)
#   #dev.off()
# }
# 
# for (coherence in c('0', '3', '10')) {
#  for (metric in c('error_consistency')) {
#    print(paste("Plotting Eidolon-experiment", metric, "for coherence", coherence))
#    plt_leg = FALSE
#    plot_any_exp('eidolon', metric = metric, use_sums =TRUE,
#                 plot_legend=plt_leg, coherence=coherence)
#    #dev.off()
#  }
# }
# 
# #dev.off()


############################################
###      Confusion Matrices
############################################

# print("Plotting TF_sums confusion for noise")
# plot_any_exp('noise', metric = 'confusion', use_sums = TRUE,
#             plot_legend=FALSE)
