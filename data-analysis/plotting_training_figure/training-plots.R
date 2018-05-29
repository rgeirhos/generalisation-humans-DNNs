###################################################################
#   Plotting script for network training experiments   
#   -------------------------------------------------------
#   Author:   Robert Geirhos
#   Based on: R version 3.2.3
###################################################################

DATAPATH = "../../raw-data/TF/"
TRAINING.DATA.PATH = "../../raw-data/fine-tuning/"

source("training-plots-helper.R")

###################################################################
#     CREATE EVALUATION vs TRAINED MODEL PLOT: FUNCTIONS
###################################################################

# number of decimal places to show within plot
NUM.DECIMAL.PLACES = 1

ALL.NETWORKS = c("sixteen01v4",
                 "sixteen03__uniform_noise_multiple",
                 "sixteen07__grayscale_contrast_multiple",
                 "sixteen08__low_pass_multiple",
                 "sixteen09__rotation_multiple",
                 "sixteen11__high_pass_multiple",
                 "sixteen16__phase_scrambling_multiple",
                 "sixteen18__color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__uniform_noise__200epochs",
                 "sixteen19__color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__salt_and_pepper_noise__200epochs",
                 "sixteen21__salt_and_pepper_noise_multiple",
                 "sixteen22__salt_and_pepper_noise_multiple__uniform_noise_multiple__200epochs",
                 "sixteen26__grayscale",
                 "sixteen27__uniform_noise_multiple__200epochs",
                 "sixteen30__color__uniform_noise_multiple__200epochs",
                 "sixteen31__grayscale__uniform_noise_multiple__200epochs",
                 "sixteen32__grayscale_contrast_multiple__uniform_noise_multiple__200epochs",
                 "sixteen33__low_pass_multiple__uniform_noise_multiple__200epochs",
                 "sixteen34__high_pass_multiple__uniform_noise_multiple__200epochs",
                 "sixteen35__phase_scrambling_multiple__uniform_noise_multiple__200epochs",
                 "sixteen36__rotation_multiple__uniform_noise_multiple__200epochs")

ALL.NETWORKS = gsub("_", "-", ALL.NETWORKS)
ALL.EXPERIMENTS = sort(list.files(TRAINING.DATA.PATH))

if("colour-png-experiment" %in% ALL.EXPERIMENTS) {
  # for colour-experiment: we would like to see both grayscale and colour
  # in the plot. Hence, a 'colour-only-exeriment' is created here
  # (for 'colour' condition)
  ALL.EXPERIMENTS = c("colour-only-experiment", ALL.EXPERIMENTS)
}

get.relevant.condition = function(experiment.name) {
  # for every experiment: determine the condition to plot.
  # They were chosen since they are conditions of an intermediate
  # difficulty: neither perfect performance nor chance-level;
  # these are the conditions for which human observers were closest
  # to an accuracy of 50%.
  if(grepl("contrast", experiment.name)) {
    return("c05")
  } else if(grepl("salt", experiment.name)) {
    # this is for salt-and-pepper noise
    return("0.2")
  } else if(grepl("noise", experiment.name)) {
    # this is for uniform noise
    return("0.35")
  } else if(grepl("rotation", experiment.name)) {
    return("90")
  } else if(grepl("highpass", experiment.name)) {
    return("0.7")
  } else if(grepl("lowpass", experiment.name)) {
    return("7")
  } else if(grepl("phase-scrambling", experiment.name)) {
    return("90")
  } else if(grepl("colour", experiment.name)) {
    if(grepl("colour-only", experiment.name)) {
      return("cr")
    } else {
      return("bw")
    }
  } else {
    stop(paste("experiment name", experiment.name, "unknown"))
  }
}

get.human.expt.data = function(expt, datapath = DATAPATH) {
  # for human observers: get datafiles without -png-
  expt = gsub("-png-", "-", expt)
  dat = get.expt.data(expt, datapath = datapath,
                      only.humans = TRUE)
  return(dat)
}

get.all.data = function(networks, experiments, datapath,
                        verbose = FALSE) {
  # Read in the relevant data for all experiments, networks
  # and also human observers. Return concise dataframe.
  
  train.conditions = c()
  eval.conditions = c()
  total.acc = c()
  
  expt.counter = 1
  for(expt in experiments) {
    if(verbose) {
      print(paste("Reading data for experiment: ", expt))
    } else {
      print(paste("Reading data for experiment #", expt.counter, "of", length(experiments)))
    }
    
    expt.data.name = expt
    if(grepl("colour-only", expt)) {
      expt.data.name = "colour-png-experiment"
    }
    
    dat = get.expt.data(expt.data.name, datapath=datapath)
    
    # determine condition for experiment
    relevant.condition = get.relevant.condition(experiment.name = expt)
    
    # add human accuracy
    if(expt != "salt-and-pepper-png-experiment") {
      human.dat = get.human.expt.data(expt.data.name)
      acc.human = get.accuracy(human.dat)
      specific.acc.human = acc.human[acc.human$x==relevant.condition, ]$y
      specific.acc.human = round(specific.acc.human, NUM.DECIMAL.PLACES)
    } else {
      # since there is, unfortunately, no human data for
      # salt-and-pepper noise, plot 'NA' instead.
      specific.acc.human = NA
    }
    if(verbose) {
      print(paste("human accuracy:", specific.acc.human))
    }
    train.conditions = c(train.conditions, "human observers")
    eval.conditions = c(eval.conditions, expt)
    total.acc = c(total.acc, specific.acc.human)
    
    # add network accuracy
    for(net in networks) {
      if(verbose) {
        print(paste(expt, net, sep=" | "))
      }
      
      # step 1: get data for specific subject
      d = dat[dat$subj == net, ]
      
      # step 2: get accuracy from specific subject data
      acc = get.accuracy(d)
      
      # step 3: filter certain condition
      if(! relevant.condition %in% unique(d$condition)) {
        stop(paste("condition not found:", relevant.condition, expt, net))
      }
      # get accuracy for relevant condition
      specific.acc = acc[acc$x==relevant.condition, ]$y
      
      # round to desired precision
      specific.acc = round(specific.acc, NUM.DECIMAL.PLACES)
      if(verbose) {
        print(specific.acc)
      }
      train.conditions = c(train.conditions, net)
      eval.conditions = c(eval.conditions, expt)
      total.acc = c(total.acc, specific.acc)
    }
    expt.counter = expt.counter + 1
  }
  if(verbose) {
    print(unique(eval.conditions))
  }
  res = data.frame(train=train.conditions, eval=eval.conditions,
                   acc=total.acc)
  return(res)
}


custom.formatting = function(dat) {
  # After data is read-in: format properly (re-name cumbersome model names etc.)
  
  dat$eval = as.character(dat$eval)
  dat[dat$eval=="contrast-png-experiment", ]$eval = "contrast (5%)"
  dat[dat$eval=="noise-png-experiment", ]$eval = "uniform noise (0.35)"
  dat[dat$eval=="colour-png-experiment", ]$eval = "greyscale"
  dat[dat$eval=="colour-only-experiment", ]$eval = "colour"
  dat[dat$eval=="rotation-experiment", ]$eval = "rotation (90°)"
  dat[dat$eval=="lowpass-experiment", ]$eval = "low-pass (std=7)"
  dat[dat$eval=="highpass-experiment", ]$eval = "high-pass (std=0.7)"
  dat[dat$eval=="phase-scrambling-experiment", ]$eval = "phase scrambling (90°)"
  dat[dat$eval=="salt-and-pepper-png-experiment", ]$eval = "salt-and-pepper noise (0.2)"
  
  dat$eval = as.factor(dat$eval)
  # order evaluation conditions in a meaningful order (not merely alphabetical)
  dat$eval = factor(dat$eval, levels = levels(dat$eval)[c(1,3,2,5,4,6,7,8,9)])
  dat$eval = factor(dat$eval, levels = rev(levels(dat$eval)))

  # later important to determine automatically which conditions were part of
  # training: save original model name
  dat$original.name = dat$train
  
  dat$train = as.character(dat$train)
  split.network.name = TRUE
  if(split.network.name) {
    # reduce network name to sixteenXX (where XX is a number), get rid of cumbersome rest
    for(i in 1:nrow(dat)) {
      dat[i, ]$train = strsplit(dat[i, ]$train, "--")[[1]][1]
    }
  }
  
  # single distortion
  dat[dat$train=="sixteen01v4", ]$train = "A1"
  dat[dat$train=="sixteen26", ]$train = "A2"
  dat[dat$train=="sixteen07", ]$train = "A3"
  dat[dat$train=="sixteen08", ]$train = "A4"
  dat[dat$train=="sixteen11", ]$train = "A5"
  dat[dat$train=="sixteen16", ]$train = "A6"
  dat[dat$train=="sixteen09", ]$train = "A7"
  dat[dat$train=="sixteen21", ]$train = "A8"
  dat[dat$train=="sixteen03", ]$train = "A9"
  
  # single distortion + uniform noise
  dat[dat$train=="sixteen30", ]$train = "B1"
  dat[dat$train=="sixteen31", ]$train = "B2"
  dat[dat$train=="sixteen32", ]$train = "B3"
  dat[dat$train=="sixteen33", ]$train = "B4"
  dat[dat$train=="sixteen34", ]$train = "B5"
  dat[dat$train=="sixteen35", ]$train = "B6"
  dat[dat$train=="sixteen36", ]$train = "B7"
  dat[dat$train=="sixteen22", ]$train = "B8"
  dat[dat$train=="sixteen27", ]$train = "B9"
  
  # multiple distortions
  dat[dat$train=="sixteen19", ]$train = "C1"
  dat[dat$train=="sixteen18", ]$train = "C2"
  
  dat$train = as.factor(dat$train)
  
  # human observers should show up in 1st column, not last -> switch
  dat$train = factor(dat$train,
                     levels = levels(dat$train)[c(length(levels(dat$train)), 1:(length(levels(dat$train))-1))])
  return(dat)
}


annotate.from.position = function(xval, yval,
                                  colour=human.100,
                                  alpha = 0.0) {
  # Helper function for plotting:
  # Return annotation rectangle from xval and yval.
  # Used in plot for drawing red rectangle around conditions
  # present in the training data.
  # xval: in [1, ..., N]
  # yval: in [1, ..., M]
  offset.x = 0.5
  offset.y = 0.5
  return(annotate("rect", xmin = offset.x +(xval-1), xmax = offset.x+xval,
                  ymin = offset.y + (yval-1), ymax = offset.y + yval,
                  alpha = alpha, colour=colour, size=1.5))
}

annotate.line.from.position = function(xval, yval,
                                       colour="black",
                                       alpha = 1.0) {
  # Helper function for plotting:
  # Return annotation line from xval and yval: used for drawing line below number
  # if 'greyscale' was part of training data since it is part of some distortions
  # xval: in [1, ..., N]
  # yval: in [1, ..., M]
  offset.x = 0.5
  margin.x = 0.125 # how much smaller than cell the line should be
  offset.y = 0.75
  return(annotate("segment", x = offset.x +(xval-1) + margin.x, xend = offset.x + xval - margin.x,
                  y = offset.y + (yval-1), yend = offset.y + (yval-1),
                  alpha = alpha, colour=colour, size=0.5))
}

annotate.vertical.line = function(xval=1) {
  # Return vertical annotation line. Used to separate
  # blocks of models from each other (e.g. visual
  # separation between humans and deep neural networks)
  offset = 0.5
  return(annotate("segment", x = xval+offset, xend = xval+offset,
                  y = 1-offset, yend = 10-offset,
                  alpha = 1.0, colour="black", size=1.7))
}


get.multiple.annotation.layers = function(g, dat, plot.vertical.lines = TRUE) {
  # Return tile with all annotation rectangles for the plot (around training conditions)
  
  if(length(unique(dat$original.name)) != length(unique(levels(dat$original.name)))) {
    stop("length must be equal: no duplicate models!")
  }
  if(length(unique(dat$eval)) != length(unique(levels(dat$eval)))) {
    stop("length must be equal: no duplicate evaluation conditions!")
  }
  
  tile = ggplot() + g
  if(plot.vertical.lines) {
    # the positions of the vertical lines are hard-coded
    tile = tile + annotate.vertical.line(xval = 1)
    tile = tile + annotate.vertical.line(xval = 10)
    tile = tile + annotate.vertical.line(xval = 19)
  }
  
  model.counter = 1
  for(model.full.name in levels(dat$train)) {
    # get actual model name (as used in training) from model
    model = as.character(unique(dat[dat$train==model.full.name, ]$original.name))
    
    eval.counter = 1
    for(e in levels(dat$eval)) {
      
      eval.condition = strsplit(e, " ")[[1]][1] # get first word
      if(eval.condition=="greyscale" && !"grayscale" %in% strsplit(model, "--")[[1]]) { # draw line
        relevant.conditions = c("contrast", "low", "high", "phase", "rotation")
        for(c in relevant.conditions) {
          if(grepl(c, model, ignore.case = TRUE)) {
            tile = tile + annotate.line.from.position(xval = model.counter, yval = eval.counter)
          }
        }
      } else { # draw rectangle
        if(grepl(eval.condition, model, ignore.case = TRUE)){
          tile = tile + annotate.from.position(xval = model.counter, yval = eval.counter)
        } else if(eval.condition == "greyscale") {
          if("grayscale" %in% strsplit(model, "--")[[1]]) {
            tile = tile + annotate.from.position(xval = model.counter, yval = eval.counter)
          }
        } else if(eval.condition == "colour") {
          model.split = strsplit(model, "--")[[1]]
          if(("color" %in% model.split) | ("vanilla ResNet-50" == model) | ("sixteen01v4" == model)) {
            tile = tile + annotate.from.position(xval = model.counter, yval = eval.counter)
          }
        }
      }
      
      eval.counter = eval.counter + 1
    }
    model.counter = model.counter + 1
  }
  return(tile)
}


plot.grid = function(dat, mark.training.conditions = TRUE,
                     plot.legend=FALSE) {
  # Create model vs. evaluation condition plot
  
  if(plot.legend) {
    scale = scale_fill_gradient(low="white", high=vgg.100,
                                name="Accuracy (%)", breaks=c(25,50,75,100),labels=c(25,50,75,100),
                                limits = c(min(dat$acc), 100))
  } else {
    scale = scale_fill_gradient(low="white", high=vgg.100, guide=FALSE)
  }
  
  g = geom_tile(aes(x=train, y=eval, fill=acc),
                data=dat, color="black", size=0.5)
  if(mark.training.conditions) {
    tile = get.multiple.annotation.layers(g, dat=dat)
  } else {
    tile = ggplot() + g
  }
  
  text.size = 15
  tile = tile +
    labs(x="Model", y="Evaluation condition") + 
    geom_text(aes(x=train, y=eval, label=sprintf("%.1f", acc)),
              data=dat, size=6, colour="black") +
    scale + 
    # formatting
    theme(axis.text.x = element_text(size=text.size+2, angle=45, colour = "black", hjust = 1),
          axis.text.y = element_text(size=text.size+2, angle=0, colour = "black"),
          axis.title.x = element_text(face="bold", size=text.size+3),
          axis.title.y = element_text(face="bold", size=text.size+3),
          legend.title = element_text(size=text.size),
          axis.ticks = element_blank(),
          panel.grid.major = element_blank(),
          panel.border = element_blank()) +
    coord_equal() # squares instead of rectangles: keep size ratio
  return(tile)
}

###################################################################
#     CREATE EVALUATION vs TRAINED MODEL PLOT: PLOTTING
###################################################################

# read data (this may take a while, in the order of 5-10 mins)
dat = get.all.data(networks = ALL.NETWORKS,
                   experiments = ALL.EXPERIMENTS,
                   datapath = TRAINING.DATA.PATH)
save.dat = dat

# format data
dat = custom.formatting(save.dat)

# look at plot
plot.grid(dat)

# print plot to file
pdf(file="../../figures/training/all_vs_all_horizontal.pdf", width=15, height=7.0)
par(mfrow=c(1,1))
plot.grid(dat)
dev.off()
