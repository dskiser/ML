path1 <- read.table("path.arff", sep = ",", header = F, comment.char = "@")

df <- as.data.frame(path1)
library(ggplot2)
ggplot(df, aes(x = df[,1], y = df[,2])) + geom_point()+ geom_path(aes(color = 1:1000)) +
  labs(title = "Intrinsic State Values", x = "", y = "")


