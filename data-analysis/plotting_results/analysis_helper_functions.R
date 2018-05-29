###################################################################
#               some general settings
###################################################################

NETWORKS = sort(c("resnet152", "googlenet", "vgg19"))
NUM.OVERALL.PARTICIPANTS = 42 # arbitrary but large enough

###################################################################
#               loading & preprocessing experimental data
###################################################################

get.expt.data = function(expt.name, onlyDNNs=FALSE, onlyHumans=FALSE) {
  # Read data and return in the correct format
  # for incompatible csv-files for DNNs and humans
  # onlyDNNs --> Loads only DNN data
  # onlyHumans --> Loads only data for humans
  
  if (onlyDNNs & onlyHumans) {
    stop("Can't plot only DNNs and only humans at the same time.")
  }
  
  if(!exists("DATAPATH")) {
    stop("you need to define the DATAPATH variable")
  }
  
  dat = NULL
  expt.path = paste(DATAPATH, expt.name, sep="")
  files = list.files(expt.path)
  if (onlyDNNs) {
    if (finetuning) {files = list.files(expt.path, "experiment_sixteen01v4|experiment_all-noise|experiment_specialised")}
    else if (useTF) {files = list.files(expt.path, "experiment_[gvr]")}
    else {files = list.files(expt.path, "experiment_[gva]")}
  } else if (onlyHumans) {
    expt.path = paste(datapath, 'humans/', expt.name, sep="")
    files = list.files(expt.path, "experiment_[s]")
  }
  
  if(length(files) < 1) {
    warning(paste("No data for expt", expt.name, "found! Check DATAPATH."))
  }
  
  for (i in 1:length(files)) {
    if(!endsWith(files[i], ".csv")) {
      warning("File without .csv ending found (and ignored)!")
    } else {
      dat = rbind(dat, read.csv(paste(expt.path, files[i], sep="/")))
    }
  }
  dat$imagename = as.character(dat$imagename)
  dat$is.correct = as.character(dat$object_response) == as.character(dat$category)
  dat$is.human = ifelse(grepl("subject", dat$subj), TRUE, FALSE)
  if (onlyDNNs  && !finetuning) {
    if (is.null(dat$object_response_sums)) {
      warning("No additional data for DNNs available")
    } else {
      dat$is.correct_sums = as.character(dat$object_response_sums) == as.character(dat$category)
    }
  }
  
  return(data.frame(experiment.name = expt.name, dat))
}

###################################################################
#               confusion plotting
###################################################################

library(ggplot2)

confusion.matrix = function(dat, subject=NULL, main=NULL, plot.scale=TRUE,
                            plot.x.y.labels=TRUE) {
  #Plot confusion matrix either for all or for a specific subject
  
  confusion = get.confusion(dat, subject)
  return(plot.confusion(confusion, unique(dat$experiment.name), subject,
                        main=main, plot.scale=plot.scale,
                        plot.x.y.labels = plot.x.y.labels))
}

get.confusion = function(dat, subject=NULL,
                         net.dat=NULL, human.dat=NULL) {
  # Sure you want to get confused? ;)
  # Return all data necessary to plot confusion matrix.
  
  if(is.null(subject)) {
    d = data.frame(dat$category,
                   dat$object_response)
  } else {
    d = data.frame(dat[dat$subj==subject, ]$category,
                   dat[dat$subj==subject, ]$object_response)
  }
  
  names(d) = c("category", "object_response") 
  
  category = as.data.frame(table(d$category))
  names(category) = c("category","CategoryFreq")
  
  confusion = as.data.frame(table(d$category, d$object_response))
  names(confusion) = c("category", "object_response", "Freq")
  
  confusion = merge(confusion, category, by=c("category"))
  confusion$Percent = confusion$Freq/confusion$CategoryFreq*100
  
  # make sure the order is correct, with 'na' in the end
  for(f in rev(c("airplane", "bear", "bicycle", "bird", "boat", "bottle",
                 "car", "cat", "chair", "clock", "dog", "elephant",
                 "keyboard", "knife", "oven", "truck", "na"))) {
    confusion$object_response <- relevel(confusion$object_response, f)
  }
  
  return(confusion)
}

plot.confusion = function(confusion, 
                          experiment.name,
                          subject=NULL,
                          is.difference.plot=FALSE,
                          main=NULL,
                          plot.accuracies=TRUE,
                          plot.x.y.labels=TRUE,
                          plot.scale = TRUE,
                          network.name=NULL) {
  # Plot confusion matrix
  
  if(is.difference.plot) {
    g = geom_tile(aes(x=category, y=object_response, fill=z),
                  data=confusion, color="black", size=0.1)
  } else {
    g = geom_tile(aes(x=category, y=object_response, fill=Percent),
                  data=confusion, color="black", size=0.1)
  }
  
  tile <- ggplot() + g +
    labs(x="presented category",y="response") + 
    if(is.null(main)) {
      ggtitle(paste("Confusion matrix", experiment.name)) 
    } else {
      ggtitle(main)
    }
  
  # print accuracy; fill gradient
  if(plot.accuracies) {
    tile = tile + 
      geom_text(aes(x=category, y=object_response, label=sprintf("%.1f", Percent)),
                data=confusion, size=8.5, colour="black")
  }
  
  tile = tile +
    if((!is.null(confusion$z)) & !is.difference.plot) {
      if(is.null(network.name)) {
        stop("no network name, but confusion$z exists -> which color to use?")
      }
      
      net.cols = NULL
      if(network.name == "vgg19") {
        net.cols = vgg19.cols
      } else if (network.name == "resnet152") {
        net.cols = resnet152.cols
      } else if (network.name == "googlenet") {
        net.cols = googlenet.cols
      }
      scale_fill_manual(values = c("0" = rgb(230, 230, 230, maxColorValue = 255),
                                   human.cols, net.cols))
    } else if(is.difference.plot) {
      print("plotting difference matrix")
      scale_fill_manual(values = c("0" = rgb(127, 127, 127, maxColorValue = 255),
                                   confdiff.human.cols, confdiff.net.cols), guide=FALSE)
    } else {
      if(plot.scale) {
        scale_fill_gradient(low=rgb(250, 250, 250, maxColorValue = 255),
                            high=human.100,
                            limits=c(0,100))
      } else {
        scale_fill_gradient(low="grey", high=human.100, guide=FALSE, limits=c(0,100))
      }
    }
  
  tile = tile + 
    geom_tile(aes(x=category, y=object_response),
              data=subset(confusion, as.character(category)==as.character(object_response)),
              color="black",size=0.3, fill="black", alpha=0) 
  if(! plot.x.y.labels) {
    tile = tile +
      theme(axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank(),
            axis.title.y=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks.y=element_blank())
  }
  return(tile)
}


###################################################################
#               helper functions
###################################################################

endsWith <- function(argument, match, ignore.case = TRUE) {
  # Return: does 'argument' end with 'match'?
  # Code adapted from:
  # http://stackoverflow.com/questions/31467732/does-r-have-function-startswith-or-endswith-like-python
  
  if(ignore.case) {
    argument = tolower(argument)
    match = tolower(match)
  }
  n = nchar(match)
  
  length = nchar(argument)
  
  return(substr(argument, pmax(1, length - n + 1), length) == match)
}


get.z.for.binomial = function(conf, conf1, conf2,
                              divide.alpha.by) {
  # Assign values within [-3, 3] indicating the 'significance color'
  # for a confusion difference plot (here, these color values are called z)
  #
  # Parameters:
  # - conf            -> confusion difference
  # - conf1           -> human confusion data
  # - conf2           -> network confusion data
  # - divide.alpha.by -> if > 1.0, Bonferroni correction will be applied
  #
  # z values:
  # -3 to -1 -> difference significant for alpha = 0.001, 0.01, 0.05; network more frequently
  # 0        -> no or no significant difference
  # 3 to 1   -> difference significant for alpha = 0.001, 0.01, 0.05; humans more frequently
  # These alpha values (0.001, 0.01, 0.05) are subject to a Bonferroni
  # correction if divide.alpha.by is assigned a value larger than 1.0
  
  conf$z = "0" # default value
  
  conf1$Freq = as.numeric(conf1$Freq)
  conf1$CategoryFreq = as.numeric(conf1$CategoryFreq)
  conf2$Freq = as.numeric(conf2$Freq)
  conf2$CategoryFreq = as.numeric(conf2$CategoryFreq)
  
  for(i in 1:nrow(conf1)) {
    if(conf1[i, ]$category != conf2[i, ]$category) {
      stop("category mismatch")
    }
    tmp = 0
    weight = 3
    for(alpha in sort(c(0.001, 0.01, 0.05), decreasing = F)) {
      val = is.in.CI(conf2[i, ]$Freq, conf2[i, ]$CategoryFreq,
                     conf1[i, ]$Freq, conf1[i, ]$CategoryFreq,
                     conf.level = 1.0-alpha/divide.alpha.by)
      if(abs(weight*val) > abs(tmp)) {
        tmp = weight*val
        break # shortcut: speed up computation and begin with most significant
      }
      weight = weight - 1
    }
    conf[i, ]$z = as.character(tmp)
  }
  return(conf)
}


is.in.CI = function(a.num.successes, a.total,
                    b.num.successes, b.total,
                    conf.level,
                    default.for.p.equals.0 = 0.001) {
  # In this analysis, is it used as follows:
  # a: network (in general, reference)
  # b: human
  #
  # Return value will be 1 if b.num.successes / b.total larger than 
  # the CI's upper bound, -1 if it is smaller, and 0 otherwise
  # (i.e. if it is contained in the CI, the return value will be 0).
  
  p.a = a.num.successes / a.total
  p.b = b.num.successes / b.total
  
  p = ifelse(p.a != 0, ifelse(p.a != 1, p.a, 1-default.for.p.equals.0), default.for.p.equals.0)
  
  p.value = binom.test(b.num.successes, b.total,
                       p = p,
                       alternative = "two.sided",
                       conf.level = conf.level)$p.value
  
  if(p.value < (1.0 - conf.level)) {
    if(p.a > p.b) {
      return(-1)
    } else if (p.b > p.a) {
      return(1)
    } else {
      stop("this shouldn't occur!")
    }
  } else {
    return(0)
  }
}
