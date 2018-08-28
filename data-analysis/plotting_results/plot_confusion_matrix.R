############################################
###       Plotting Confuion Matrix
###
###  Function to plot all confusion matrices
###  All parameters were set in plot_any_exp()
############################################

plot_confusion_matrix = function() {
  
  # define output file 
  framework <- '_TF'
  
  if (use_sums) {extension <- '_sums'} 
  else {extension <- ''}
  
  if (exp_name == 'eidolon') {
    extension = paste('_coh_', coherence, extension, sep = "")
  }
  if (finetuning) {
    extension = paste(extension, '_finetuned', sep = '')
  }
  
  par(mfrow=c(1,1), cex.axis=tick_sizes, mar=c(6.1, 4.7, 4.1, 2.1), omi=c(0,0,0,0))
  
  subjs <- as.character(unique(expdata$subj))
  for (subject in c("humans", subjs)) {
    for (cond in conds) {
      
      cond_ <- gsub("\\.", "", toString(cond))
      pdf(file=paste(confusionpath, exp_name, "_confusion_", subject, "_c", cond_, extension, framework, ".pdf", sep=""),
      # pdf(file=paste('../../../error_figures/confusion/', exp_name, "_confusion_", subject, "_", cond, extension, framework, ".pdf", sep=""),
          width=16,
          height=17)

      # plot confusion matrices
      if (subject == "humans") {
        p <- confusion.matrix(expdata[expdata$condition==cond & expdata$is.human==TRUE, ],
                              plot.x.y.labels = FALSE,
                              plot.scale = FALSE,
                              main="")
      } else {
        p <- confusion.matrix(expdata[expdata$condition==cond & expdata$subj==subject, ],
                              plot.x.y.labels = FALSE,
                              plot.scale = FALSE,
                              main="")
      }
      print(p)
      while (dev.cur() != 1) {
        dev.off()
      }
    }
  }

}
