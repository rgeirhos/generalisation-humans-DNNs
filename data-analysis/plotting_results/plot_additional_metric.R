############################################
###       Plotting metric
###
###  Function to plot all other metrics
### for the paper only entropy1000 was used
### however there are more options like
### entropy16, ...
### All parameters were set in plot_any_exp()
### in conjuction with plotting_parameters.py
############################################

plot_additional_metric = function () {
  
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
  
  pdf(file=paste(folder_path, exp_name, "_", metric, extension, framework, ".pdf", sep=""),
      width=7.5, 
      height=6.5)
  
  par(mfrow=c(1,1), cex.axis=tick_sizes, mar=c(6.1, 4.7, 4.1, 2.1))
  
  # just create a framework for the plot
  # x, y should fall outside xlim and ylim
  plot(1000, 1000, xlim = xlims, ylim = c(y_lower_lim, y_upper_lim), main=main,
       ylab = ylabel, xlab = x_label, axes= FALSE, cex.lab = axis_label_sizes)
  
  if (plot_legend) {
    # add legend
    if (typeof(legend_pos) == "character") {
      legend(legend_pos, legend = legend_subj, lty=1,
             col = plot_colours, box.lty = legend_box_lty,
             cex = legend_size, pch = point_types, seg.len = legend_line_length,
             pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1)
    } else {
      print(legend_pos[1])
      print(legend_pos[2])
      legend(legend_pos[1], legend_pos[2],
             legend = legend_subj, lty=1,
             col = plot_colours, box.lty = legend_box_lty,
             cex = legend_size, pch = point_types, seg.len = legend_line_length,
             pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1)
    }
  }
  
  # check if the metric requires a comparison line
  if (startsWith(metric, "entropy_16")) {
    abline(h=2.52, lty=3)
  }
  if (startsWith(metric, "entropy_1000")) {
    abline(h=9.97, lty=3)
  }
  if (startsWith(metric, "is.correct")) {
    abline(h = 1/16, lty=3)
  }
  
  # plot x-axis:
  axis(1, at=ticks, labels=x_tick_labels)
  
  # plot y-axis
  axis(2, at=seq(from=y_lower_lim, to=y_upper_lim, by=y_axis_steps))
  
  # iterate over subjects
  for (subject in subjects) {
    
    # get data for subject
    subj_data <- expdata[expdata$subj == subject,]
    
    # average over all but conditions and subjects
    mean_metric_by_sess <- aggregate(metric ~ condition * session, data = subj_data, FUN = mean)
    
    # average over subjects
    mean_metric <- aggregate(metric ~ condition, data = mean_metric_by_sess, FUN = mean)
    mean_metric <- mean_metric[rank(conds),]
    
    print(subject)
    print(mean_metric)
    
    # get point coordinates to plot
    # plot data points (first get their coordinates)
    x_coords = x_values
    y_coords = mean_metric$metric
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
    n <- length(unique(mean_metric_by_sess$session))
    stds <- aggregate(metric ~ condition, data = mean_metric_by_sess, FUN = sd)
    stds <- stds[rank(conds),]         
    sem <- stds$metric / sqrt(n)
    sem <- sem[rank(conds)]        
    if (error_bars == "SD") {
      errors <- stds$metric
      y0 <- mean_metric$metric - errors
      y1 <- mean_metric$metric + errors
    }
    if (error_bars == "SE") {
      errors <- sem
      y0 <- mean_metric$metric - errors
      y1 <- mean_metric$metric + errors
    }
    if (error_bars == "range") {
      y0 = aggregate(metric ~ condition, data = mean_metric_by_sess, FUN = min)$metric
      y0 <- y0[rank(conds)]         
      y1 = aggregate(metric ~ condition, data = mean_metric_by_sess, FUN = max)$metric
      y1 <- y1[rank(conds)]    
    }
    
    # plot error bars
    # hack: we draw arrows but with very special "arrowheads"
    if (!normalise) {
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
    }
    
    # plotting separated points
    if (separate_left) {
      points(x_values[1], mean_metric$metric[1], type = "b", 
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
      points(x_values[length(x_values)], mean_metric$metric[length(x_values)], type = "b", 
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
  
  dev.off()
  
}