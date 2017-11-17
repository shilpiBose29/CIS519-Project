# install.packages("rARPACK")
library(logisticPCA)
library(ggplot2)
amn <- read_csv("datasets/Asheville/amn.csv")
logpca_model = logisticPCA(amn, k = 16, m = 10)
write.table(logpca_model$PCs, "datasets/Asheville/amn_PCAed.csv", 
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

