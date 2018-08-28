library(fields)
source("analysis_helper_functions.R")
confusionpath <- "../../figures/confusion/"

pdf(file=paste(confusionpath, "colorbar.pdf", sep=""),
    width=10,
    height=100)

par(mar=c(3,35,3,0))
colpal = colorRampPalette(c(confusion_0_prec, human.100))
z=matrix(1:100,nrow=1)
x=1
y=seq(0,100,len=100) 
image(x, y, z, col=colpal(100), axes=FALSE, xlab="",ylab="", xaxt='n', yaxt='n')
axis(side=2, las=1, cex.axis=15, tick=FALSE,
     labels=c("0%", "25%", "50%", "75%", "100%"), at=c(0, 25, 50, 75, 100))

dev.off()
