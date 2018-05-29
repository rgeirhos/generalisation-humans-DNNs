############################################
###       Plotting Homing
###
###  Function to plot all homing plots
### All parameters were set in plot_any_exp()
### in conjuction with plotting_parameters.py
############################################

plot_homing = function() {
  
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
  plot(1000, 1000, xlim = xlims, ylim = c(0, 4), main=main,
       ylab = y_label_hom, xlab = x_label, axes= FALSE, cex.lab = axis_label_sizes)
  
  # add line indicating entropy for uniform distribution
  abline(h = 4, lty=3)
  
  # plot x axis
  axis(1, at=ticks, labels=x_tick_labels)
  
  # plot y-axis
  axis(2, at=seq(from=0, to=4, by=0.5))
  
  if (plot_legend) {
    # add legend
    legend(legend_pos_hom, legend = legend_subj, lty=1,
           col = plot_colours, box.lty = legend_box_lty, cex = legend_size, pch = point_types, seg.len = legend_line_length,
           pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1, bg = "transparent")
  }
  
  # iterate over subjects
  for (subject in subjects) {
    
    # get data for subject
    if (subject == "humans") {subj_data <- expdata[expdata$is.human == TRUE,]}
    else {subj_data <- expdata[expdata$subj == subject,]}
    
    # add a coloum for counting responses
    subj_data$count <- 1
    
    # check how many sessions there are
    if (subject == "humans") {num_sess <- length(unique(subj_data$subj))}
    else {num_sess <- length(unique(subj_data$session))}
    
    # count occurences of response-category pairs
    num_responses_by_condition <- aggregate(count ~ condition * object_response, data = subj_data, FUN = sum)
    num_responses_by_condition <- num_responses_by_condition[num_responses_by_condition$object_response != "na",]
    
    # count how many times each response was given in a particular condition
    nr_conds <- length(conds)
    response_counts = vector(mode = "list", length = nr_conds)
    names(response_counts) <- conds
    for (i in 1:nr_conds) {
      cond = conds[i]
      cond <- toString(cond)
      response_counts[[i]] <- num_responses_by_condition[num_responses_by_condition$condition == cond,]
    }
    
    # calculate entropy by condition
    entropy = rep(0, nr_conds)
    total_nr_trials_per_cond <- num_sess * num_trials / nr_conds
    for (i in 1:nr_conds) {
      cond <- toString(conds[i])
      response_count <- response_counts[[cond]]$count
      total_nr_responses <- sum(response_count)
      for (cell in response_count) {
        prob <- cell / total_nr_responses
        if (prob != 0) {
          entropy[i] = entropy[i] - prob * log2(prob)
        }
      }
    }
    
    # get point coordinates to plot
    # plot data points (first get there coordinates)
    x_coords = x_values
    y_coords = entropy
    if (separate_left) {
      x_coords = x_coords[2:length(x_values)]
      y_coords = y_coords[2:length(x_values)]
    } else if (separate_right) {
      x_coords = x_coords[1:length(x_values)-1]
      y_coords = y_coords[1:length(x_values)-1]
    } 
    
    print(subject)
    print(entropy)
    
    # plot points
    points(x_coords, y_coords, type = "b", pch = point_types[[subject]], lwd = line_width,
           col = plot_colours[[subject]], lty = 1, cex = point_sizes[subject])
    
    # plotting separated points
    if (separate_left) {
      points(x_values[1], entropy[1], type = "b", 
             pch = point_types[[subject]], lwd = line_width,           
             col = plot_colours[[subject]], cex = point_sizes[subject])
    } else if (separate_right) {
      points(x_values[length(x_values)], entropy[length(x_values)], type = "b", 
             pch = point_types[[subject]], lwd = line_width,         
             col = plot_colours[[subject]], cex = point_sizes[subject])
    }
  }
  
  
  dev.off()
  
}
