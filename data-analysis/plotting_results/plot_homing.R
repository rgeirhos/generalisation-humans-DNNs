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
  #pdf(file=paste('../../../error_figures/Homing_', exp_name, "_", metric, extension, framework, ".pdf", sep=""),
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
    if (subject == "humans") {num_responses_by_condition_by_sess <- aggregate(count ~ condition * object_response * subj, data = subj_data, FUN = sum)}
    else {num_responses_by_condition_by_sess <- aggregate(count ~ condition * object_response * session, data = subj_data, FUN = sum)}
    num_responses_by_condition_by_sess <- num_responses_by_condition_by_sess[num_responses_by_condition_by_sess$object_response != "na",]
    
    # calculate entropies for each session individually
    entropies_by_sess <- data.frame()
    
    if (subject == "humans") {sessions <- unique(num_responses_by_condition_by_sess$subj)}
    else {sessions <- unique(num_responses_by_condition_by_sess$session)}
    
    for (sess in sessions) {
      # count how many times each response was given in a particular condition
      if (subject == "humans") {
        num_responses_sess <- num_responses_by_condition_by_sess[num_responses_by_condition_by_sess$subj == sess,]
      }
      else {
        num_responses_sess <- num_responses_by_condition_by_sess[num_responses_by_condition_by_sess$session == sess,]
      }
      nr_conds <- length(conds)
      response_counts = vector(mode = "list", length = nr_conds)
      names(response_counts) <- conds
      for (i in 1:nr_conds) {
        cond = conds[i]
        cond <- toString(cond)
        response_counts[[i]] <- num_responses_sess[num_responses_sess$condition == cond,]
      }
      
      # calculate entropy by condition
      entropy = rep(0, nr_conds)
      for (i in 1:nr_conds) {
        cond <- toString(conds[i])
        response_count <- response_counts[[cond]]$count
        total_nr_responses <- sum(response_count)
        prob_sum <- 0
        cnt <- 0
        for (cell in response_count) {
          prob <- cell / total_nr_responses
          prob_sum <- prob_sum + prob
          cnt <- cnt + 1
          if (prob != 0) {
            entropy[i] = entropy[i] - prob * log2(prob)
          }
        }
        new_row <- list(sess, cond, entropy[i])
        names(new_row) <- list("session", "condition", "entropy")
        entropies_by_sess <- rbind(entropies_by_sess, new_row)
        levels(entropies_by_sess$session) <- sessions
        levels(entropies_by_sess$condition) <- conds
      }
    }
    mean_entropies <- aggregate(entropy ~ condition, data = entropies_by_sess, FUN = mean)
    
    # get point coordinates to plot
    # plot data points (first get there coordinates)
    x_coords = x_values
    y_coords = mean_entropies$entropy
    if (separate_left) {
      x_coords = x_coords[2:length(x_values)]
      y_coords = y_coords[2:length(x_values)]
    } else if (separate_right) {
      x_coords = x_coords[1:length(x_values)-1]
      y_coords = y_coords[1:length(x_values)-1]
    } 
    
    print(subject)
    print(mean_entropies$entropy)
    
    # plot points
    points(x_coords, y_coords, type = "b", pch = point_types[[subject]], lwd = line_width,
           col = plot_colours[[subject]], lty = 1, cex = point_sizes[subject])
    
    # draw errorbars
    # calculate standard deviation / standard error of the mean / response range
    # then get y coordinates for error bars
    if (subject == "humans") {n <- length(unique(entropies_by_sess$subj))}
    else {n <- length(unique(entropies_by_sess$session))}
    stds <- aggregate(entropy ~ condition, data = entropies_by_sess, FUN = sd)
    #stds <- stds[rank(conds),]        
    sem <- stds$entropy / sqrt(n)
    sem <- sem[rank(conds)]         
    if (error_bars == "SD") {
      errors <- stds$entropy
      y0 <- mean_entropies$entropy - errors
      y1 <- mean_entropies$entropy + errors
    }
    if (error_bars == "SE") {
      errors <- sem
      y0 <- mean_entropies$entropy - errors
      y1 <- mean_entropies$entropy + errors
    }
    if (error_bars == "range") {
      y0 <- aggregate(entropy ~ condition, data = entropies_by_sess, FUN = min)$entropy
      #y0 <- y0[rank(conds)]       
      y1 <- aggregate(entropy ~ condition, data = entropies_by_sess, FUN = max)$entropy
      #y1 <- y1[rank(conds)]         
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
    
    # plotting separated points
    if (separate_left) {
      points(x_values[1], mean_entropies$entropy[1], type = "b", 
             pch = point_types[[subject]], lwd = line_width,           
             col = plot_colours[[subject]], cex = point_sizes[subject])
    } else if (separate_right) {
      points(x_values[length(x_values)], mean_entropies$entropy[length(x_values)], type = "b", 
             pch = point_types[[subject]], lwd = line_width,         
             col = plot_colours[[subject]], cex = point_sizes[subject])
    }
  }
  
  
  dev.off()
  
}
