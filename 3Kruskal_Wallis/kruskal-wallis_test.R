# -*- coding: utf-8 -*-
# #############################################################
# Kruskal-Wallis test to analyse if difference between 8 ML and 10 ensemble methods is significant
# Author: Brigitta Szabó (szabo.brigitta@atk.hu)
# Date: 10/03/2023
# #############################################################

library(reshape)
library(agricolae)
library(plyr)
library(xlsx)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dir("predictedSSM")


# test random ----
test_r <- read.csv("/home/jovyan/R/test_random_2.csv")
str(test_r)

res <- cbind(test_r[,c("X")],(test_r$observed_SSM)-(test_r[,3:length(test_r)]))
str(res)
names(res)[2:length(res)] <- paste("res_",names(res)[2:length(res)],sep="")
names(res)[1] <- "X"
str(res)
summary(res)

data_m <- melt(res, id=(c("X")))
names(data_m)[2:3] <- c("method", "res")
str(data_m)
data_m$abs <- abs(data_m$res)
data_m$se <- (data_m$res)^2
summary(data_m)

# Kruskal-Wallis test based on squared error
Kruskal_test_r <- with(data_m, kruskal(se, method, group=TRUE, console=TRUE))
write.csv(Kruskal_test_r$groups, "Kruskal_Wallis_test_r.csv")
save(Kruskal_test_r, file = "Kruskal_Wallis_test_r.rdata")

rm(list=ls())


# test temporal ----
test_t <- read.csv("/home/jovyan/R/test_temporal_3.csv")
str(test_t)

res <- cbind(test_t[,c("X")],(test_t$observed_SSM)-(test_t[,3:length(test_t)]))
str(res)
names(res)[2:length(res)] <- paste("res_",names(res)[2:length(res)],sep="")
names(res)[1] <- "X"
str(res)
summary(res)

data_m <- melt(res, id=(c("X")))
names(data_m)[2:3] <- c("method", "res")
str(data_m)
data_m$abs <- abs(data_m$res)
data_m$se <- (data_m$res)^2
summary(data_m)

# Kruskal-Wallis test based on squared error
Kruskal_test_t <- with(data_m, kruskal(se, method, group=TRUE, console=TRUE))
write.csv(Kruskal_test_t$groups, "Kruskal_Wallis_test_t.csv")
save(Kruskal_test_t, file = "Kruskal_Wallis_test_t.rdata")

rm(list=ls())

# test independent stations ----
test_i <- read.csv("/home/jovyan/R/test_independent-stations_3.csv")
str(test_i)

res <- cbind(test_i[,c("X")],(test_i$observed_SSM)-(test_i[,3:length(test_i)]))
str(res)
names(res)[2:length(res)] <- paste("res_",names(res)[2:length(res)],sep="")
names(res)[1] <- "X"
str(res)
summary(res)

data_m <- melt(res, id=(c("X")))
names(data_m)[2:3] <- c("method", "res")
str(data_m)
data_m$abs <- abs(data_m$res)
data_m$se <- (data_m$res)^2
summary(data_m)

# Kruskal-Wallis test based on squared error
Kruskal_test_i <- with(data_m, kruskal(se, method, group=TRUE, console=TRUE))
write.csv(Kruskal_test_i$groups, "Kruskal_Wallis_test_i.csv")
save(Kruskal_test_i, file = "Kruskal_Wallis_test_i.rdata")

hist(data_m$abs)
hist(data_m$se)
hist(data_m$res)


# format into tables ----
load("Kruskal_Wallis_test_r.rdata")
Kruskal_test_r$means$methods <- rownames(Kruskal_test_r$means)
Kruskal_t_r <- merge(Kruskal_test_r$means[, c(1:2, 10)], Kruskal_test_r$groups, by.x = "rank", by.y = "se")
Kruskal_t_r

load("Kruskal_Wallis_test_t.rdata")
Kruskal_test_t$means$methods <- rownames(Kruskal_test_t$means)
Kruskal_t_t <- merge(Kruskal_test_t$means[, c(1:2, 10)], Kruskal_test_t$groups, by.x = "rank", by.y = "se")
Kruskal_t_t

load("Kruskal_Wallis_test_i.rdata")
Kruskal_test_i$means$methods <- rownames(Kruskal_test_i$means)
Kruskal_t_i <- merge(Kruskal_test_i$means[, c(1:2, 10)], Kruskal_test_i$groups, by.x = "rank", by.y = "se")
Kruskal_t_i

Kruskal_all <- join_all(list(Kruskal_t_i, Kruskal_t_t, Kruskal_t_r), by = 'methods')
#Kruskal_all <- join_all(list(Kruskal_t_r), by = 'methods')
Kruskal_all
write.csv(Kruskal_all, "Kruskal_all.csv")
write.xlsx(Kruskal_all[, c(3, 1,2,4,5:10)], "Kruskal_all.xlsx", append = F, row.names = F)
