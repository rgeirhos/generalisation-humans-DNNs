############################################
###       Plotting Accuracy
###
###  Function to plot all accuracy plots
### All parameters were set in plot_any_exp()
### in conjuction with plotting_parameters.py
############################################

plot_accuracy = function() {
  
  # define output file name
  framework <- '_TF'

  if (use_sums) {extension <- '_sums'} 
  else {extension <- ''}
  
  if (exp_name == 'eidolon') {
    extension = paste('_coh_', coherence, extension, sep = "")
  }
  if (finetuning) {
    extension = paste(extension, '_finetuned', sep = '')
  }
  
  pdf(file=paste(folder_path, exp_name, "_", metric, extension, framework, ".pdf", sep=""),
      width=7.5, 
      height=6.5)
  
  par(mfrow=c(1,1), cex.axis=tick_sizes, mar=c(6.1, 4.7, 4.1, 2.1))
  
  # just create a framework for the plot
  # x, y should fall outside xlim and ylim
  plot(1000, 1000, xlim = xlims, ylim = c(y_lower_lim, y_upper_lim), main=main,
       ylab = y_label_acc, xlab = x_label, axes= FALSE, cex.lab = axis_label_sizes) 
  
  # add line indicating chance performance
  abline(h = 1/16, lty=3)
  
  # plot x-axis:
  axis(1, at=ticks, labels=x_tick_labels)
  
  # plot y-axis
  axis(2, at=seq(from=y_lower_lim, to=1, by=y_axis_steps))
  
  if (plot_legend) {
    # add legend
    if (typeof(legend_pos_acc) == "character") {
      legend(legend_pos_acc, legend = legend_subj, lty=1,
             col = plot_colours, box.lty = legend_box_lty,
             cex = legend_size, pch = point_types, seg.len = legend_line_length,
             pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1)
    } else {
      print(legend_pos_acc[1])
      print(legend_pos_acc[2])
      legend(legend_pos_acc[1], legend_pos_acc[2],
             legend = legend_subj, lty=1,
             col = plot_colours, box.lty = legend_box_lty,
             cex = legend_size, pch = point_types, seg.len = legend_line_length,
             pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1)
    }
  }
  
  # iterate over subjects and collect mean accuracies for raw-accuracy csv files
  raw_accuracies <- data.frame(conds)
  if (length(conds) == length(x_tick_labels)) {
    raw_accuracies['condition'] <- x_tick_labels
  } else {
    raw_accuracies['condition'] <- conds
  }
  raw_accuracies$conds <- NULL
  
  for (subject in subjects) {
    
    # get data for subject
    if (subject == "humans") {subj_data <- expdata[expdata$is.human == TRUE,]}
    else {subj_data <- expdata[expdata$subj == subject,]}
    
    # average over all but conditions and subjects and
    if (subject == "humans") {
      mean_accuracies_by_sess <- aggregate(is.correct ~ condition * subj, data = subj_data, FUN = mean)
    }
    else {
      mean_accuracies_by_sess <- aggregate(is.correct ~ condition * session, data = subj_data, FUN = mean)
    }
    
    # average over subjects / reorder if conds' order is different to r's inherent ordering (rank())
    mean_accuracies <- aggregate(is.correct ~ condition, data = mean_accuracies_by_sess, FUN = mean)
    mean_accuracies <- mean_accuracies[rank(conds),]     
    
    # output subject and accuracies + save accuracies into csv
    #print(subject)
    #print(mean_accuracies)
    raw_accuracies[subject] = mean_accuracies$is.correct
    
    # get point coordinates to plot
    # plot data points (first get there coordinates)
    x_coords <- x_values           
    y_coords = mean_accuracies$is.correct 
    if (separate_left) {
      x_coords = x_coords[2:length(x_values)]
      y_coords = y_coords[2:length(x_values)]
    } else if (separate_right) {
      x_coords = x_coords[1:length(x_values)-1]
      y_coords = y_coords[1:length(x_values)-1]
    } 
    
    # plot points
    points(x_coords, y_coords, type = "b", pch = point_types[[subject]], lwd = line_width,
           col = plot_colours[[subject]], lty = line_type, cex = point_sizes[subject])
    
    # draw errorbars
    # calculate standard deviation / standard error of the mean / response range
    # then get y coordinates for error bars
    if (subject == "humans") {n <- length(unique(mean_accuracies_by_sess$subj))}
    else {n <- length(unique(mean_accuracies_by_sess$session))}
    stds <- aggregate(is.correct ~ condition, data = mean_accuracies_by_sess, FUN = sd)
    stds <- stds[rank(conds),]        
    sem <- stds$is.correct / sqrt(n)
    sem <- sem[rank(conds)]         
    if (error_bars == "SD") {
      errors <- stds$is.correct
      y0 <- mean_accuracies$is.correct - errors
      y1 <- mean_accuracies$is.correct + errors
    }
    if (error_bars == "SE") {
      errors <- sem
      y0 <- mean_accuracies$is.correct - errors
      y1 <- mean_accuracies$is.correct + errors
    }
    if (error_bars == "range") {
      y0 <- aggregate(is.correct ~ condition, data = mean_accuracies_by_sess, FUN = min)$is.correct
      y0 <- y0[rank(conds)]       
      y1 <- aggregate(is.correct ~ condition, data = mean_accuracies_by_sess, FUN = max)$is.correct
      y1 <- y1[rank(conds)]         
    }
    
    # plot error bars
    # hack: we draw arrows but with very special "arrowheads"
    if (separate_left) {
      arrows(x0=x_coords, y0=y0[2:length(x_values)],
             x1=x_coords, y1=y1[2:length(x_values)],
             length=0.025, angle=90, code=3,
             col=plot_colours[[subject]],
             lwd = line_width)
    } else if (separate_right) {
      arrows(x0=x_coords, y0=y0[1:length(x_values)-1],
             x1=x_coords, y1=y1[1:length(x_values)-1],
             length=0.025, angle=90, code=3,
             col=plot_colours[[subject]],
             lwd = line_width)
    } else {
      arrows(x0=x_coords, y0=y0,
             x1=x_coords, y1=y1,
             length=0.025, angle=90, code=3,
             col=plot_colours[[subject]],
             lwd = line_width)
    }
    
    # plotting separated points (e.g 0 in log plot)
    if (separate_left) {
      points(x_values[1], mean_accuracies$is.correct[1], type = "b", 
             pch = point_types[[subject]], lwd = line_width,            
             col = plot_colours[[subject]], cex = point_sizes[subject])
      if (!normalise) {
        arrows(x0=x_values[1], y0=y0[1],
               x1=x_values[1], y1=y1[1],
               length=0.025, angle=90, code=3,
               col=plot_colours[[subject]],
               lwd = line_width)
      }
    } else if (separate_right) {
      points(x_values[length(x_values)], mean_accuracies$is.correct[length(x_values)], type = "b", 
             pch = point_types[[subject]], lwd = line_width,          
             col = plot_colours[[subject]], cex = point_sizes[subject])
      if (!normalise) {
        arrows(x0=x_values[length(x_values)], y0=y0[length(x_values)],
               x1=x_values[length(x_values)], y1=y1[length(x_values)],
               length=0.025, angle=90, code=3,
               col=plot_colours[[subject]],
               lwd = line_width)
      }
    }
    
  }
  
  # write accuracies to csv in raw-accuracies folder
  print(raw_accuracies)
  write.csv(raw_accuracies,
            file = paste(acc_path, exp_name, "_accuracies", extension, framework, ".csv", sep=""),
            row.names=FALSE)
  
  dev.off()
  
}