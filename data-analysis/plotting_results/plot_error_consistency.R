############################################
###       Plotting Homing
###
###  Function to plot all homing plots
### All parameters were set in plot_any_exp()
### in conjuction with plotting_parameters.py
############################################

plot_error_consistency = function() {
  
  line_names <- c("Humans-Humans",
                  "Humans-GoogLeNet",
                  "Humans-VGG19",
                  "Humans-ResNet152",
                  "DNNs-DNNs")
  
  legend_subj <- line_names
  
  ###########--> parameters for whole plot
  
  y_lower_lim <- 0
  #y_upper_lim <- 2
  y_upper_lim <- 0.15
  #y_axis_steps <- 0.2
  y_axis_steps <- 0.025
  #y_label <- "Jenson-Shannon-Div. [bits]"  
  y_label <- "L2-Norm of error distributions"
  legend_pos_error <- "topright"
  
  ###########--> parameters for individual lines
  # define colours for participants
  plot_colours <- c(rgb(165, 30, 55, maxColorValue = 255),  
                    rgb(80, 170, 200, maxColorValue = 255),
                    rgb(0, 105, 170, maxColorValue = 255),
                    rgb(65, 90, 140, maxColorValue = 255),
                    rgb(0, 0, 0, maxColorValue = 255))
  names(plot_colours) <- line_names
  
  # define what point type to use for which participant
  point_types <- c(1, 15, 17, 18, 19)
  names(point_types) <- line_names
  
  # define what point type to use for which participant
  point_sizes <- c(point_size, point_size, point_size, 1.4*point_size, point_size)
  names(point_sizes) <- line_names
  
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
  
  pdf(file=paste('../../../error_figures/L2_', exp_name, "_", metric, extension, framework, ".pdf", sep=""),
      #pdf(file=paste('../../../error_figures/', exp_name, "_", metric, extension, framework, ".pdf", sep=""),
      #pdf(file=paste(folder_path, exp_name, "_", metric, extension, framework, ".pdf", sep=""),
      width=7.5, 
      height=6.5)
  
  par(mfrow=c(1,1), cex.axis=tick_sizes, mar=c(6.1, 4.7, 4.1, 2.1))
  
  # just create a framework for the plot
  # x, y should fall outside xlim and ylim
  plot(1000, 1000, xlim = xlims, ylim = c(y_lower_lim, y_upper_lim), main=main,
       ylab = y_label, xlab = x_label, axes= FALSE, cex.lab = axis_label_sizes)
  
  # plot x axis
  axis(1, at=ticks, labels=x_tick_labels)
  
  # plot y-axis
  axis(2, at=seq(from=y_lower_lim, to=y_upper_lim, by=y_axis_steps))
  
  if (plot_legend) {
    # add legend
    legend(legend_pos_error, legend = legend_subj, lty=1,
           col = plot_colours, box.lty = legend_box_lty, cex = legend_size, pch = point_types, seg.len = legend_line_length,
           pt.cex = point_sizes, lwd = line_width, y.intersp = 1, x.intersp = 1, bg = "transparent")
  }
  
  #################
  # Calculate JSDs / L2
  #################
  
  kld = function(P, Q) {
    eps <- 1 / 10000
    s <- 0
    for (i in 1:length(P)) {
      p = P[i]
      q = Q[i]
      if (p == 0) {
        p <- eps
      } 
      if (q == 0) {
        q <- eps
      }
      s = s + p * log2(p/q)
    }
    return(s)
  }
  
  jsd = function(P, Q) {
   # first normalise
   P <- P / sum(P)
   Q <- Q / sum(Q)
   # calculate KL divergences
   return(0.5 * (kld(P, Q) + kld(Q,P)))
  }
  
  l2 = function(P, Q) {
    P <- P / sum(P)
    Q <- Q / sum(Q)
    return(sum(abs(P-Q)^2))
  }
  
  #TODO check that all conditions, subjects, classes have some number of rows
  expdata$is.false <- 1 - expdata$is.correct
  df_by_class <- aggregate(is.false ~ condition * subj * category,
                           data = expdata[expdata$object_response != 'na',], FUN = mean)
  #View(df_by_class)
  count_by_class <- aggregate(is.false ~ condition * subj * category, data = expdata, FUN = length)
  #View(count_by_class)
  
  ################
  # Plot
  ################
  
  subj_names <- unique(df_by_class$subj)
  human_names <- c()
  dnn_names <- c()
  for (subj_name in subj_names) {
    if (startsWith(subj_name, "subject")) {
      human_names <- c(human_names, subj_name)
    } else {
      dnn_names <- c(dnn_names, subj_name)
    }
  }
  print(human_names)
  print(dnn_names)
  
  # iterate over subjects
  for (name in line_names) {
    
    y_coords = c()
    if (name == "DNNs-DNNs") {
      for (cond in conds) {
        l2_mean <- 0
        cnt <- 0
        for (i in 1:(length(dnn_names)-1)) {
          for (j in (i+1):length(dnn_names)) {
            probs_subj <- df_by_class$is.false[(df_by_class$subj == dnn_names[i]) &
                                                 (df_by_class$condition == cond)]
            probs_subj2 <- df_by_class$is.false[(df_by_class$subj == dnn_names[j]) &
                                                  (df_by_class$condition == cond)]
            # calculate L2
            l2_mean <- l2_mean + l2(probs_subj, probs_subj2)
            cnt <- cnt + 1
          }
        }
        l2_mean <- l2_mean / cnt
        y_coords <- c(y_coords, l2_mean)
      }
    } else if (name == "Humans-Humans") {
      print('H-H')
      for (cond in conds) {
        l2_mean <- 0
        cnt <- 0
        for (i in 1:(length(human_names)-1)) {
          for (j in (i+1):length(human_names)) {
            probs_subj <- df_by_class$is.false[(df_by_class$subj == human_names[i]) &
                                                 (df_by_class$condition == cond)]
            probs_subj2 <- df_by_class$is.false[(df_by_class$subj == human_names[j]) &
                                                  (df_by_class$condition == cond)]
            # calculate L2
            l2_mean <- l2_mean + l2(probs_subj, probs_subj2)
            print(l2_mean)
            cnt <- cnt + 1
          }
        }
        print(cnt)
        l2_mean <- l2_mean / cnt
        y_coords <- c(y_coords, l2_mean)
      }
    } else {
      for (cond in conds) {
        l2_mean <- 0
        cnt <- 0
        for (human_name in human_names) {
          # get dnn name
          if (name == "Humans-GoogLeNet") {
            dnn_name <- "googlenet"
          } else if (name == "Humans-ResNet152") {
            dnn_name <- "resnet152"
          } else if (name == "Humans-VGG19") {
            dnn_name <- "vgg19"
          }
          # retrieve probabilities of errors
          probs_subj <- df_by_class$is.false[(df_by_class$subj == human_name) &
                                               (df_by_class$condition == cond)]
          probs_subj2 <- df_by_class$is.false[(df_by_class$subj == dnn_name) &
                                                (df_by_class$condition == cond)]
          # calculate L2
          l2_mean <- l2_mean + l2(probs_subj, probs_subj2)
          cnt <- cnt + 1
        }
        l2_mean <- l2_mean / cnt
        y_coords <- c(y_coords, l2_mean)
      }
    }
    
    # get point coordinates to plot
    # plot data points (first get there coordinates)
    x_coords = x_values
    if (separate_left) {
      x_coords = x_coords[2:length(x_values)]
      y_coords = y_coords[2:length(x_values)]
    } else if (separate_right) {
      x_coords = x_coords[1:length(x_values)-1]
      y_coords = y_coords[1:length(x_values)-1]
    }
    
    print(name)
    print(y_coords)
    
    # plot points
    points(x_coords, y_coords, type = "b", pch = point_types[[name]], lwd = line_width,
           col = plot_colours[[name]], lty = 1, cex = point_sizes[name])
    
    # plotting separated points
    if (separate_left) {
      points(x_values[1], y_coords[1], type = "b",
             pch = point_types[[name]], lwd = line_width,
             col = plot_colours[[name]], cex = point_sizes[name])
    } else if (separate_right) {
      points(x_values[length(x_values)], y_coords[length(x_values)], type = "b",
             pch = point_types[[name]], lwd = line_width,
             col = plot_colours[[name]], cex = point_sizes[name])
    }
  }
  
  dev.off()
}
