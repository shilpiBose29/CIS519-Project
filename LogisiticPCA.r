# install.packages("rARPACK")
library(logisticPCA)
library(ggplot2)
amn <- read_csv("datasets/Asheville/amn.csv")
logpca_model = logisticPCA(amn, k = 16, m = 10)
write.table(logpca_model$PCs, "datasets/Asheville/amn_PCAed.csv", 
            row.names = FALSE, col.names = FALSE, sep=',')

write.table(logpca_model$U, "datasets/Asheville/amn_PCAed_U.csv", 
            row.names = FALSE, col.names = FALSE, sep=',')

# clogpca_model = convexLogisticPCA(amn, k = 16, m = 10)
#logpca_cv = cv.lpca(house_votes84, ks = 2:16, ms = 1:10, quiet = FALSE)#
#plot(logpca_cv)

#logpca_k16_cv = cv.lpca(house_votes84, ks = 16, ms = 1:60, quiet = FALSE)#
#plot(logpca_k16_cv)

#logpca_k2_cv = cv.lpca(house_votes84, ks = 2, ms = 1:20, quiet = FALSE)#
#plot(logpca_k2_cv)

#logpca_k8_cv = cv.lpca(house_votes84, ks = 8, ms = 1:20, quiet = FALSE)#
#plot(logpca_k8_cv)

df = as.data.frame(logpca_model$U)

df$idu <- as.numeric(row.names(df))

library(reshape2)
df_melten <- melt(df, id.vars="idu")

df_melten$value = abs(df_melten$value)

ggplot(df_melten, aes(idu, value, col=variable)) + geom_line() 

df_melten_thresheld = df_melten[(df_melten$value>.3),]

ggplot(df_melten_thresheld, aes(idu, value)) + 
   geom_line() +
   facet_wrap(~variable)
library(dplyr)
group_by(df_melten_thresheld, variable)



