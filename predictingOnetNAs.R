# Load libraries
library(Matrix)
library(reshape2)
library(recommenderlab)
library(FNN)
library(kohonen)

# subset dataset for just the O*Net survey columns
ds <- read.csv("analysisDS.csv", header=T, stringsAsFactors=F)
ds_onet <- ds[,c(1,2,56515:56861)]

## Run some EDA on data

# Count of NA by variable
countVarNAs <- function(x) sum(is.na(x))
varNA.cnt.col <- sapply(ds_onet, countVarNAs)
varNA.cnt.row <- apply(ds_onet, 1, countVarNAs)
which(varNA.cnt.row > 0)
length(which(varNA.cnt.row > 0))

# Occupations with any NAs present
ds_onet[which(varNA.cnt.row > 0), 2]

# Occupations with all NAs present
ds_onet[which(varNA.cnt.row >= 347), 2]


# Nearest Neighbor Search

# Transpose ngram matrix to apply SOM classification on
ds_ngrams <- ds[, c(11:56514)]
ds_ngram.tr <- t(ds_ngrams)
colNames <- t(ds[,2])
colnames(ds_ngram.tr) <- colNames

# Convert untransposed matrix of ngrams to numerical matrix representation in R
ds_ngrams.mat <- data.matrix(ds_ngrams)
rownames(ds_ngrams.mat) = ds_onet[,2]

# Apply the SOM Classification on untransposed dataset 
# NOTE: This will only group occupations in terms of their ngrams
system.time(ngrams.som.0505 <- som(ds_ngrams.mat, grid=somgrid(5,6, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.0510 <- som(ds_ngrams.mat, grid=somgrid(5,10, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.1005 <- som(ds_ngrams.mat, grid=somgrid(10,5, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.1010 <- som(ds_ngrams.mat, grid=somgrid(10,10, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.1015 <- som(ds_ngrams.mat, grid=somgrid(10,15, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.1510 <- som(ds_ngrams.mat, grid=somgrid(15,10, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.1020 <- som(ds_ngrams.mat, grid=somgrid(10,20, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")

system.time(ngrams.som.2010 <- som(ds_ngrams.mat, grid=somgrid(20,10, "hexagonal")))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")


# Apply the SOM Classification on transposed dataset 
# NOTE: This will only group ngrams in terms of the occupations

ngrams.som.0505.t <- som(ds_ngram.tr, grid=somgrid(5,5))
ngrams.som.0510.t <- som(ds_ngram.tr, grid=somgrid(5,10))
ngrams.som.1010.t <- som(ds_ngram.tr, grid=somgrid(10,10))
ngrams.som.1520.t <- som(ds_ngram.tr, grid=somgrid(15,20))
ngrams.som.2020.t <- som(ds_ngram.tr, grid=somgrid(20,20))
save.image("/Volumes//Badr/Plytos//analytics/m1/ngramSOM.RData")