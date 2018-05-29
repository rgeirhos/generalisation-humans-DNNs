#################################################################################
###       Plotting Parameters
#
# Contains a list of predefined plotting parameters for different
# experiments and subjects
################################################################################

############--> for all experiments
# General parameters, that apply accros experiments
error_bars <- "range" # options are: "SD", "SE", "range" 
axis_label_sizes <- 2.4
tick_sizes <- 2
legend_line_length <- 2
legend_size <- 2.15
if (finetuning) {legend_subj <- c("participants (avg.)", "ResNet-50", "All-Distortions-Net", "Specialised-Net")
} else if (additional) {legend_subj <- c("GoogLeNet", "VGG-19", "ResNet-152")
} else {legend_subj <- c("participants (avg.)", "GoogLeNet", "VGG-19", "ResNet-152")}
legend_box_lty <- 1     
line_width <- 3
line_type <- 1
point_size <- 3

##############--> for current metric
# Specify parameters depending on metric
y_label_acc <- "Classification accuracy"         
y_label_hom <- "Response distr. entropy [bits]" 
y_lower_lim <- 0
y_upper_lim <- 1
y_axis_steps <- 0.1
if (startsWith(metric, "entropy_16")) {y_upper_lim <- 4; y_axis_steps <- 0.5}
if (startsWith(metric, "entropy_1000")) {y_upper_lim <- 10; y_axis_steps <- 2}
if (startsWith(metric, "entropy")) {
  ylabel <- "Probability vector entropy [bits]"  
} else if (startsWith(metric, "is.correct")) {
  ylabel <- "Classification accuracy" 
} else {ylabel <- metric}

############--> for individual experiments (second batch)
    
if (exp_name == "false-colour") {
  num_trials <- 1280
  conds <- c("true", "false")
  xlims <- c(0.5, 2.5)
  ticks <-  c(1, 2)
  x_values <- ticks               
  log_base <- 1
  legend_pos_acc <- c(0.5+0.27*2, 1/16+0.35) #"bottom"
  legend_pos_hom <- "bottom"
  legend_pos <- c(xlims[1], 9.97) #"topright"
  x_label = "Colour"
  x_tick_labels = c("true", "opponent")
}
if (startsWith(exp_name, "highpass")) {
  num_trials <- 1280
  conds <- c("Inf", 3, 1.5, 1.0, 0.7, 0.55, 0.45, 0.4)
  xlims <- c(4, 0.4)
  ticks <-  c(4, 3, 1.5, 1.0, 0.7, 0.55, 0.45, 0.4)
  x_values <- ticks                     
  log_base <- 2
  legend_pos_acc <- "topright"
  legend_pos_hom <- "bottomleft"
  legend_pos <- "bottom"
  x_label = "Filter standard deviation"
  x_tick_labels = conds
}
if (exp_name == "lowpass") {
  num_trials <- 1280
  conds <- c(0, 1, 3, 5, 7, 10, 15, 40)
  xlims <- c(0, 40)
  ticks <-c(0, 1, 3, 5, 7, 10 , 15, 40)
  x_values <- ticks                   
  log_base <- 2
  legend_pos_acc <- "topright"
  legend_pos_hom <- "bottomleft"
  legend_pos <- c(xlims[1], 9.97) # "topleft"
  x_label = "Filter standard deviation"
  x_tick_labels = conds
}
if (exp_name == "phase-scrambling") {
  num_trials <- 1280
  conds <- c(0, 30, 60, 90, 120, 150, 180)
  xlims <- c(0, 180)
  ticks <-  c(0, 30, 60, 90, 120, 150, 180)
  x_values <- ticks                   
  log_base <- 1
  legend_pos_acc <- "topright"
  legend_pos_hom <- "bottomleft"
  legend_pos <- c(xlims[1], 9.97) #"topleft"
  x_label = "Phase noise width [°]"
  x_tick_labels = conds
}
if (exp_name == "power-equalisation") {
  num_trials <- 1280
  conds <- c("0", "pow")
  xlims <- c(0.5, 2.5)
  ticks <-  c(1, 2)
  x_values <- ticks                   
  log_base <- 1
  legend_pos_acc <- c(0.5+0.27*2, 1/16+0.35) #"bottom"
  legend_pos_hom <- "bottom"
  legend_pos <- c(xlims[1], 9.97) # "topleft"
  x_label = "Power spectrum"
  x_tick_labels = c("original", "equalised")
}
if (exp_name == "rotation") {
  num_trials <- 1280
  conds <- c(0, 90, 180, 270)
  xlims <- c(0, 270)
  ticks <-  conds
  x_values <- ticks                    
  log_base <- 1
  legend_pos_acc <- c(0.5+0.27*270, 1/16+0.35) #"bottom"
  legend_pos_hom <- "bottom"
  legend_pos <- c(xlims[1], 9.97) # "topleft"
  x_label = "Rotation angle [°]"
  x_tick_labels = conds
}

# for individual experiments (first batch)
if (startsWith(exp_name, "colour")) { 
  num_trials <- 1280
  conds <- c("cr", "bw")
  xlims <- c(0.5, 2.5)
  ticks <-  c(1, 2)
  x_values <- ticks                     
  log_base <- 1
  legend_pos_acc <- "center"#c(0.5+0.27*2, 1/16+0.45) #c(0.5+0.27*2, 1/16+0.35) #"bottom"
  legend_pos_hom <- "bottom"
  legend_pos <- c(xlims[1], 9.97) # "top"
  x_label = "Colour"
  x_tick_labels = c("colour", "greyscale") 
}

if (startsWith(exp_name, "contrast")) {
  num_trials <- 1280
  conds <- c('c100', 'c50', 'c30', 'c15', 'c10', 'c05', 'c03', 'c01')
  xlims <- c(100, 1)
  ticks <-  c(100, 10^1.5, 10, 10^0.5, 1)
  x_values <- c(100, 50, 30, 15, 10, 5, 3, 1)     
  log_base <- 10
  legend_pos_acc <- "topright"
  legend_pos_hom <- "bottomleft"
  legend_pos <- c(xlims[1], 9.97) #"topleft"
  x_label = expression("Log"[10]*" of contrast in percent")
  x_tick_labels = c(2.0, 1.5, 1.0, 0.5, 0)
}

if (startsWith(exp_name, "noise")) {
  num_trials <- 1280
  conds <- c(0, 0.03, 0.05, 0.1, 0.2, 0.35, 0.6, 0.9)
  xlims <- c(0, 1)
  ticks <-  c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  x_values <- conds    
  log_base <- 1
  legend_pos_acc <- "topright"
  legend_pos_hom <- "right"
  legend_pos <- c(0.67, 9.97) # "topright"
  x_label = 'Uniform noise width'
  x_tick_labels = ticks
}

if (startsWith(exp_name, "salt-and-pepper-png")) {
  num_trials <- 1280
  conds <- c(0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
  xlims <- c(0, 100)
  ticks <-  c(0, 0.2, 0.4, 0.6, 0.8, 1.0) * 100
  x_values <- conds * 100   
  log_base <- 1
  legend_pos_acc <- "topright"
  legend_pos_hom <- "right"
  legend_pos <- c(0.67, 9.97) # "topright"
  x_label = 'Salt and pepper noise probability [%]'
  x_tick_labels = ticks
  legend_subj <- legend_subj[2:4]
}

if (startsWith(exp_name, "eidolon")) {
  num_trials <- 1280
  conds <- c('1', '2', '4', '8', '16', '32', '64', '128')
  xlims <- c(1, 128)
  ticks <-  c(1, 2, 4, 8, 16, 32, 64, 128)
  x_values <- ticks    
  log_base <- 2
  legend_pos_acc <- "topright"
  legend_pos_hom <- "bottomleft"
  if (coherence == '10') {legend_pos <- c(xlims[1], 9.97)}# "topleft"}
  else {legend_pos <- "bottom"}
  x_label = expression("Log"[2]*" of 'reach' parameter")
  x_tick_labels = c(0, 1, 2, 3, 4, 5, 6, 7)
  
  # filter out coherence values that aren't used and adjust conditions (conds) accordingly
  expdata <- expdata[endsWith(expdata$condition, (paste('-', coherence, '-10', sep = ""))),]
  for (i in 1:length(conds)) {
    conds[i] <- paste(conds[i], '-', coherence, '-10', sep='')
  }
}

###########--> for subjects
# define colours for participants
if (!additional && (exp_name != 'salt-and-pepper-png')) {
  plot_colours <- c(rgb(165, 30, 55, maxColorValue = 255),  
                    rgb(80, 170, 200, maxColorValue = 255),
                    rgb(0, 105, 170, maxColorValue = 255),
                    rgb(65, 90, 140, maxColorValue = 255))
  # take new colours for finetuning plots
  if (finetuning) {
    plot_colours <- c(rgb(165, 30, 55, maxColorValue = 255),  
                      rgb(43, 140, 190, maxColorValue = 255),
                      rgb(123, 204, 196, maxColorValue = 255),
                      rgb(186, 228, 188, maxColorValue = 255))
  }
} else {
  plot_colours <- c(rgb(80, 170, 200, maxColorValue = 255),
                    rgb(0, 105, 170, maxColorValue = 255),
                    rgb(65, 90, 140, maxColorValue = 255))
  # take new colours for finetuning plots
  if (finetuning) {
    plot_colours <- c(rgb(43, 140, 190, maxColorValue = 255),
                      rgb(123, 204, 196, maxColorValue = 255),
                      rgb(186, 228, 188, maxColorValue = 255))
  }
}
names(plot_colours) <- subjects

# define what point type to use for which participant
if (additional || (exp_name == 'salt-and-pepper-png')) {
  point_types <- c(15, 17, 18)
} else {
  point_types <- c(1, 15, 17, 18)
}
names(point_types) <- subjects

# define what point type to use for which participant
if (additional || (exp_name == 'salt-and-pepper-png')) {
  point_sizes <- c(point_size, point_size, 1.4*point_size)
} else {
  point_sizes <- c(point_size, point_size, point_size, 1.4*point_size)
}
names(point_sizes) <- subjects

############################################
###       Adjusting Parameters
############################################
# adjust parameters for example for logarithmic plots

# check if the left-most or right-most value has to be plotted separately
separate_left = FALSE       
separate_right = FALSE

# adjust x=inf point
if ('Inf' %in% conds) {
  
  # check if inf is on left or right interval end
  if (conds[1] == 'Inf') {
    separate_left <- TRUE
  } else if (tail(conds, 1) == 'Inf') {
    separate_right <- TRUE
  } else {
    stop('value Inf occurs within interval for logarithmic plot')
  }
  
}

# adjust x axis (especially x=0) in case of a logarithmic plot
if (log_base > 1) {
  
  # is there a zero value to correct?
  if ('0' %in% x_values) {
    
    # check of zero is on left or right interval end
    if (x_values[1] == 0) {
      separate_left <- TRUE
    } else if (tail(x_values, 1) == 0) {
      separate_right <- TRUE
    } else {
      stop('value 0 occurs within interval for logarithmic plot')
    }
    
    # determine value to replace zero
    min_value <- min(x_values[x_values != 0])
    x_value_0 <- min_value/log_base
    
    # replace zeros by dummy value
    if (xlims[1] < xlims[2]) {
      xlims <- c(x_value_0, xlims[2])
    } else {
      xlims <- c(xlims[1], x_value_0)
    }
    ticks[ticks == 0] <- x_value_0
    x_values[x_values == 0 ] <- x_value_0
    
  }
  
  # adjust axis
  ticks <- log(ticks, base=log_base)
  x_values <- log(x_values, base=log_base)
  xlims <- log(xlims, base=log_base)
  
  # also adjust absolute legend position x-value
  if (typeof(legend_pos_acc) == "double") {
    if (legend_pos_acc[1] != 0) {
      legend_pos_acc[1] = log(legend_pos_acc[1], base=log_base)
    } else {
      legend_pos_acc[1] = xlims[1]
    }
  }
  if (typeof(legend_pos) == "double") {
    if (legend_pos[1] != 0) {
      legend_pos[1] = log(legend_pos[1], base=log_base)
    } else {
      legend_pos[1] = xlims[1]
    }
  }
}
