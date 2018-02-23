library(jsonlite)
library(stringr)
library(stringdist)
library(tidyr)
library(caret)
library(sqldf)
library(syuzhet)
library(coreNLP)
library(lubridate)
library(corrplot)
library(parallel)
library(foreach)
library(doSNOW)
library(dplyr)

no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, outfile="")
registerDoSNOW(cl)

setwd("C:/Users/Bryan/Google Drive/Kaggle/TwoSigma")
set.seed(314)

# Reading data in json format
        
json_train <- "C:/Users/Bryan/Google Drive/Kaggle/TwoSigma/train.json"
extracted <- fromJSON(json_train)
json_test <- "C:/Users/Bryan/Google Drive/Kaggle/TwoSigma/test.json"
extracted_test <- fromJSON(json_test)

rm(json_train); rm(json_test)

# Creating data frame to be used for extracted/cleaned data

traindata <- as.data.frame(
                        matrix(data = NA, 
                               ncol = length(extracted), 
                               nrow = length(unlist(extracted[1]))
                               )
                        )
names(traindata) <- names(extracted)

testdata <- as.data.frame(
      matrix(data = NA, 
             ncol = length(extracted_test), 
             nrow = length(unlist(extracted_test[1]))
      )
)
names(testdata) <- names(extracted_test)

# The photos and features fields from the original data contain more data 
# and need to be cleaned prior to incorporating them into the main dataset.  The following
# script creates and executes an array of command strings that brings over all of the other
# data fields into the traindata dataframe.

en <- names(extracted)
en <- en[which(!(en %in% c("photos", "features")))]
ev <- paste(paste("traindata", en, sep = "$"), "<-", paste("extracted", en, sep = "$"), sep = "")

for (i in 1:length(ev)) {
      eval(parse(text = ev[i]))
}
rm(en);rm(ev)

en <- names(extracted_test)
en <- en[which(!(en %in% c("photos", "features")))]
ev <- paste(paste("testdata", en, sep = "$"), "<-", paste("extracted_test", en, sep = "$"), sep = "")

for (i in 1:length(ev)) {
      eval(parse(text = ev[i]))
}
rm(en);rm(ev)

# Because each variable was imported as a list, they must all be converted to the proper 
# data type

traindata$bathrooms <- as.numeric(unlist(traindata$bathrooms))
traindata$bedrooms <- as.numeric(unlist(traindata$bedrooms))
traindata$building_id <- as.factor(unlist(traindata$building_id))
traindata$created <- as.character(unlist(traindata$created))
traindata$description <- as.character(unlist(traindata$description))
traindata$display_address <- as.character(unlist(traindata$display_address))
traindata$latitude <- as.numeric(unlist(traindata$latitude))
traindata$longitude <- as.numeric(unlist(traindata$longitude))
traindata$listing_id <- as.integer(unlist(traindata$listing_id))
traindata$manager_id <- as.factor(unlist(traindata$manager_id))
traindata$price <- as.numeric(unlist(traindata$price))
traindata$street_address <- as.character(unlist(traindata$street_address))
traindata$interest_level <- as.factor(unlist(traindata$interest_level))

testdata$bathrooms <- as.numeric(unlist(testdata$bathrooms))
testdata$bedrooms <- as.numeric(unlist(testdata$bedrooms))
testdata$building_id <- as.factor(unlist(testdata$building_id))
testdata$created <- as.character(unlist(testdata$created))
testdata$description <- as.character(unlist(testdata$description))
testdata$display_address <- as.character(unlist(testdata$display_address))
testdata$latitude <- as.numeric(unlist(testdata$latitude))
testdata$longitude <- as.numeric(unlist(testdata$longitude))
testdata$listing_id <- as.integer(unlist(testdata$listing_id))
testdata$manager_id <- as.factor(unlist(testdata$manager_id))
testdata$price <- as.numeric(unlist(testdata$price))
testdata$street_address <- as.character(unlist(testdata$street_address))
testdata$interest_level <- ""

# Beginning the process to extract key features from features field
# Extracting features and normalizing their format

extracted$features <- parLapply(cl, extracted$features, 
                             function(x)
                                   list(tolower(unlist(x))))
extracted$features <- parLapply(cl, extracted$features,
                             function(x)
                                   list(gsub("[^[:lower:]=\\.]", "", unlist(x))))

extracted_test$features <- parLapply(cl, extracted_test$features, 
                                     function(x)
                                           list(tolower(unlist(x))))
extracted_test$features <- parLapply(cl, extracted_test$features,
                                     function(x)
                                           list(gsub("[^[:lower:]=\\.]", "", unlist(x))))

# Due to the storage size of the numerous pictures available, there will be no analysis 
# performed on an image by image basis.  Instead, file names will be examined and number 
# images available for each listing will be used as potential features.  
# Listing features will be analyzed in detail below.

traindata$numphotos <- parSapply(cl, extracted$photos, function(x) length(unlist(x)))
traindata$features <- parSapply(cl, extracted$features, function(x) paste(unlist(x), collapse = ","))
traindata$numfeatures <- parSapply(cl, extracted$features, function(x) length(unlist(x)))
traindata$photos <- parSapply(cl, extracted$photos, function(x) paste(unlist(x), collapse = ","))
traindata$features <- gsub("\\.", "", traindata$features)

testdata$numphotos <- parSapply(cl, extracted_test$photos, function(x) length(unlist(x)))
testdata$features <- parSapply(cl, extracted_test$features, function(x) paste(unlist(x), collapse = ","))
testdata$numfeatures <- parSapply(cl, extracted_test$features, function(x) length(unlist(x)))
testdata$photos <- parSapply(cl, extracted_test$photos, function(x) paste(unlist(x), collapse = ","))
testdata$features <- gsub("\\.", "", testdata$features)

rm(extracted); rm(extracted_test)

nofee <- grep(".*nofee.*", traindata$features)
for (i in 1:nrow(traindata)) {
      ifelse(i %in% nofee, traindata$nofee[i] <- 1, traindata$nofee[i] <- 0)
}
nofee <- grep(".*nofee.*", testdata$features)
for (i in 1:nrow(testdata)) {
      ifelse(i %in% nofee, testdata$nofee[i] <- 1, testdata$nofee[i] <- 0)
}

rm(nofee)
traindata <- select(traindata, -features)
testdata <- select(testdata, -features)

# There are over 1700 different features listed, many of which are equivalent to each other.
# The first step is to identify the most common features.

features <- as.data.frame(unlist(str_split(traindata$features, pattern = ",")))
names(features) <- c("Var1")
features <- subset(features, Var1 != "")
uniquefeatures <- as.data.frame(table(features))
names(uniquefeatures) <- c("Var1","Freq")
uniquefeatures <- uniquefeatures[order(-uniquefeatures$Freq),]
f <- data.frame(head(uniquefeatures$Var1, 100))
names(f) <- "keyfeatures"
features$Var2 <- features$Var1

foreach (i = 1:nrow(f)) %dopar% {
      features$Var2 <- gsub(paste(".*", f$keyfeatures[i], ".*", sep = ""), f$keyfeatures[i], features$Var2)
}

featurest <- as.data.frame(unlist(str_split(testdata$features, pattern = ",")))
names(featurest) <- c("Var1")
featurest <- subset(featurest, Var1 != "")
featurest$Var2 <- featurest$Var1

foreach (i = 1:nrow(f)) %dopar% {
      featurest$Var2 <- gsub(paste(".*", f$keyfeatures[i], ".*", sep = ""), f$keyfeatures[i], featurest$Var2)
}

rm(f)

# Once the common features are listed, it's beneficial to aggregate very similar features
# into those. Example:  'centralair' and 'centralac' clearly indicate the presence of
# 'airconditioning'

features$Var2 <- gsub(".*ss.*appl.*|.*stainless.*", "stainlesssteelappliances", features$Var2)
features$Var2 <- gsub(".*dryer.*|.*drier.*|.*washerinunit.*|.*dyer.*", "washerdryer", features$Var2)
features$Var2 <- gsub(".*central.*ac.*|.*centralair.*", "airconditioning", features$Var2)
features$Var2 <- gsub(".*playroom.*", "playroom", features$Var2)
features$Var2 <- gsub(".*wifi.*|.*internet.*", "highspeedinternet", features$Var2)
features$Var2 <- gsub(".*reno.*", "renovated", features$Var2)
features$Var2 <- gsub(".*deck.*", "deck", features$Var2)
features$Var2 <- gsub(".*king.*bedroom.*", "king", features$Var2)
features$Var2 <- gsub(".*queen.*bedroom.*", "queen", features$Var2)
features$Var2 <- gsub("nopet$", "nopets", features$Var2)
features$Var2 <- gsub(".*pets.*|.*petfriendly.*", "pets", features$Var2)
features$Var2 <- gsub(".*doorman.*", "concierge", features$Var2)
features$Var2 <- gsub(".*valet.*", "valet", features$Var2)

featurest$Var2 <- gsub(".*ss.*appl.*|.*stainless.*", "stainlesssteelappliances", featurest$Var2)
featurest$Var2 <- gsub(".*dryer.*|.*drier.*|.*washerinunit.*|.*dyer.*", "washerdryer", featurest$Var2)
featurest$Var2 <- gsub(".*central.*ac.*|.*centralair.*", "airconditioning", featurest$Var2)
featurest$Var2 <- gsub(".*playroom.*", "playroom", featurest$Var2)
featurest$Var2 <- gsub(".*wifi.*|.*internet.*", "highspeedinternet", featurest$Var2)
featurest$Var2 <- gsub(".*reno.*", "renovated", featurest$Var2)
featurest$Var2 <- gsub(".*deck.*", "deck", featurest$Var2)
featurest$Var2 <- gsub(".*king.*bedroom.*", "king", featurest$Var2)
featurest$Var2 <- gsub(".*queen.*bedroom.*", "queen", featurest$Var2)
featurest$Var2 <- gsub("nopet$", "nopets", featurest$Var2)
featurest$Var2 <- gsub(".*pets.*|.*petfriendly.*", "pets", featurest$Var2)
featurest$Var2 <- gsub(".*doorman.*", "concierge", featurest$Var2)
featurest$Var2 <- gsub(".*valet.*", "valet", featurest$Var2)

# Even when the simpler synonymous terms are aggregated, there are still hundreds of feature
# terms that are rarely used that could be condensed into one of the common features
# determined above

uniquefeatures <- as.data.frame(table(features$Var2))
names(uniquefeatures) <- c("Var1","Freq")
uniquefeatures <- uniquefeatures[order(-uniquefeatures$Freq),]

featuredistance <- as.data.frame(matrix(data = NA, ncol = nrow(uniquefeatures)+1, nrow = 102))
names(featuredistance) <- c("featurename", as.character(uniquefeatures$Var1))
featuredistance$featurename <- c(as.character(uniquefeatures$Var1[1:100]),
                                  as.character("MinDist"),
                                  as.character("NewFeatureName"))

uniquefeaturest <- as.data.frame(table(featurest$Var2))
names(uniquefeaturest) <- c("Var1","Freq")
uniquefeaturest <- uniquefeaturest[order(-uniquefeaturest$Freq),]

featuredistancet <- as.data.frame(matrix(data = NA, ncol = nrow(uniquefeaturest)+1, nrow = 102))
names(featuredistancet) <- c("featurename", as.character(uniquefeaturest$Var1))
featuredistancet$featurename <- c(as.character(uniquefeaturest$Var1[1:100]),
                                  as.character("MinDist"),
                                  as.character("NewFeatureName"))

featuredistancet[which(!(featuredistancet$featurename %in% featuredistance$featurename)),1] <-
      featuredistance[which(!(featuredistance$featurename %in% featuredistancet$featurename)),1]

# Creating data frame that compares the 'string distance' of all uncommon features to all
# of the common ones.  Similar to the way a spell checker might work, string distance
# effectively measures the number of characters by which two strings differ.  Once the
# string distances are populated, a new field is created that indicates which (if any)
# of the main features falls within a specified scaled distance. The rare feature is
# converted to match the main feature with the smallest scaled string distance.
# Rare features that do not have a close match are removed. The entire process condenses
# 1700 potential factor variables into 100--much more computationally manageable.

for (i in 1:100) {
      for (j in 2:ncol(featuredistance)) {
            featuredistance[i,j] <- stringdist(featuredistance[i,1], names(featuredistance)[j], method = "jaccard")
      }
}

for (i in 1:100) {
      for (j in 2:ncol(featuredistance)) {
            featuredistance[i,j] <- stringdist(featuredistance[i,1], names(featuredistance)[j], method = "jaccard")
      }
}

for (i in 2:ncol(featuredistance)) {
      featuredistance[101,i] <- min(featuredistance[1:100,i])
}

for (i in 2:ncol(featuredistance)) {
      if (featuredistance[101,i] <= 0.12) {
            featuredistance[102,i] <- featuredistance[which(featuredistance[1:100,i] == min(featuredistance[1:100,i])),][1]
      } else {
            featuredistance[102,i] <- ""
      }
}

for (i in 1:100) {
      for (j in 2:ncol(featuredistancet)) {
            featuredistancet[i,j] <- stringdist(featuredistancet[i,1], names(featuredistancet)[j], method = "jaccard")
      }
}

for (i in 1:100) {
      for (j in 2:ncol(featuredistancet)) {
            featuredistancet[i,j] <- stringdist(featuredistancet[i,1], names(featuredistancet)[j], method = "jaccard")
      }
}

for (i in 2:ncol(featuredistancet)) {
      featuredistancet[101,i] <- min(featuredistancet[1:100,i])
}

for (i in 2:ncol(featuredistancet)) {
      if (featuredistancet[101,i] <= 0.12) {
            featuredistancet[102,i] <- featuredistancet[which(featuredistancet[1:100,i] == min(featuredistancet[1:100,i])),][1]
      } else {
            featuredistancet[102,i] <- ""
      }
}


featurereplacements <- as.data.frame(matrix(data = NA, nrow = nrow(uniquefeatures), ncol = 2))
names(featurereplacements) <- c("old", "new")
featurereplacements$old <- names(featuredistance)[2:ncol(featuredistance)]
featurereplacements$new[1:ncol(featuredistance)-1] <- featuredistance[102,2:ncol(featuredistance)]

featurereplacementst <- as.data.frame(matrix(data = NA, nrow = nrow(uniquefeaturest), ncol = 2))
names(featurereplacementst) <- c("old", "new")
featurereplacementst$old <- names(featuredistancet)[2:ncol(featuredistancet)]
featurereplacementst$new[1:ncol(featuredistancet)-1] <- featuredistancet[102,2:ncol(featuredistancet)]

rm(features); rm(uniquefeatures); rm(featuredistance)
rm(featurest); rm(uniquefeaturest); rm(featuredistancet)

tempfeatures <- as.data.frame(cbind.data.frame(traindata$features, traindata$numfeatures))
names(tempfeatures) <- c("features", "qty")
tempfeatures <- separate(data = tempfeatures,
                        col = features,
                        into = paste("feature", 1:max(tempfeatures$qty), sep = ""),
                        sep = ",")

tempfeaturest <- as.data.frame(cbind.data.frame(testdata$features, testdata$numfeatures))
names(tempfeaturest) <- c("features", "qty")
tempfeaturest <- separate(data = tempfeaturest,
                          col = features,
                          into = paste("feature", 1:max(tempfeatures$qty), sep = ""),
                          sep = ",")


for (i in 1:nrow(tempfeatures)) {
      for (j in 1:ncol(tempfeatures)-1) {
            ifelse (tempfeatures[i,j] %in% featurereplacements$old,
                    tempfeatures[i,j] <- featurereplacements$new[which(featurereplacements$old == tempfeatures[i,j])],
                    tempfeatures[i,j] <- "")
      }
}

for (i in 1:nrow(tempfeaturest)) {
      for (j in 1:ncol(tempfeaturest)-1) {
            ifelse (tempfeaturest[i,j] %in% featurereplacementst$old,
                    tempfeaturest[i,j] <- featurereplacementst$new[which(featurereplacementst$old == tempfeaturest[i,j])],
                    tempfeaturest[i,j] <- "")
      }
}

f <- unique(featurereplacements$new)
f <- f[-100]

ft <- unique(featurereplacementst$new)
ft <- f[-100]

# All remaining features are converted to individual categorical variables

newf <- as.data.frame(matrix(data = NA,
                             ncol = length(f),
                             nrow = nrow(tempfeatures)))
names(newf) <- f

for (i in 1:ncol(tempfeatures)) {
      for (j in 1:nrow(tempfeatures)) {
            if (tempfeatures[j,i] %in% names(newf)) {
                  index <- which(names(newf) == tempfeatures[j,i])
                  newf[j,index] <- 1
            }
      }
}
newf[is.na(newf)] <- 0

traindata <- data.frame(traindata,newf)
rm(newf);rm(tempfeatures);rm(featurereplacements);rm(f);rm(index)

traindata <- select(traindata, -features)

newft <- as.data.frame(matrix(data = NA,
                              ncol = length(ft),
                              nrow = nrow(tempfeaturest)))
names(newft) <- ft

for (i in 1:ncol(tempfeaturest)) {
      for (j in 1:nrow(tempfeaturest)) {
            if (tempfeaturest[j,i] %in% names(newft)) {
                  index <- which(names(newft) == tempfeaturest[j,i])
                  newft[j,index] <- 1
            }
      }
}
newft[is.na(newft)] <- 0

testdata <- data.frame(testdata,newft)

rm(newft);rm(tempfeaturest);rm(featurereplacementst);rm(ft);rm(index)

testdata <- select(testdata, -features)

# There are a handful of listings where the price listed was exorbitant; most of those were
# improperly classified as rentals when they were in fact for sale.  Removed those listings
# from the dataset.  Further price analysis will follow later.

traindata <- subset(traindata, price <= 100000)
traindata$price <- log(traindata$price + 1)
testdata$price[which(testdata$price > 100000)] <- testdata$price[which(testdata$price > 100000)]/100 
testdata$price <- log(testdata$price + 1)


# Plotting map to view geographical locations of each listing

statelines <- map_data("state")
p <- ggplot()
p <- p + geom_polygon(data = statelines, aes(x = long, y = lat, group = group))
p <- p + scale_fill_discrete(guide=FALSE)
p <- p + geom_point(data = traindata, aes(x = longitude, y = latitude), color = "red")
p <- p + scale_x_continuous(limits = c(min(traindata$longitude), max(traindata$longitude)))
p <- p + scale_y_continuous(limits = c(min(traindata$latitude), max(traindata$latitude)))
plot(p)

# The mapping indicates that there are a few geographical outliers while a substantial
# number of the listings are clustered around a specific region.  Attempts to normalize 
# outlier lat/long data by crossreferencing the building_id field did not provide any 
# corrections. Outliers weren't removed, but had their lat/long readings converted to NA.

noll <- traindata$listing_id[which(traindata$latitude <= 40.55 |
                                         traindata$latitude >= 40.95 |
                                         traindata$longitude <= -74.125 |
                                         traindata$longitude >= -73.65)]
traindata$building_id <- as.factor(unlist(traindata$building_id))
unclassified <- traindata[which(traindata$listing_id %in% noll),]
unclassified <- select(unclassified, listing_id, building_id, longitude, latitude)
llb <- data.frame(traindata$listing_id, traindata$building_id)
llb <- llb[-which(llb$traindata.listing_id %in% unclassified$listing_id),]

rm(llb)

traindata$latitude[which(traindata$listing_id %in% unclassified$listing_id)] <- 0
traindata$longitude[which(traindata$listing_id %in% unclassified$listing_id)] <- 0

noll <- testdata$listing_id[which(testdata$latitude <= 40.55 |
                                        testdata$latitude >= 40.95 |
                                        testdata$longitude <= -74.125 |
                                        testdata$longitude >= -73.65)]
testdata$building_id <- as.factor(unlist(testdata$building_id))
unclassified <- testdata[which(testdata$listing_id %in% noll),]
unclassified <- select(unclassified, listing_id, building_id, longitude, latitude)
llb <- data.frame(testdata$listing_id, testdata$building_id)
llb <- llb[-which(llb$testdata.listing_id %in% unclassified$listing_id),]

rm(llb)

testdata$latitude[which(testdata$listing_id %in% unclassified$listing_id)] <- 0
testdata$longitude[which(testdata$listing_id %in% unclassified$listing_id)] <- 0

rm(unclassified); rm(noll)

# The new plot shows the locations of all listings without outliers present.
 
p <- ggplot()
p <- p + geom_polygon(data = statelines, aes(x = long, y = lat, group = group))
p <- p + scale_fill_discrete(guide=FALSE)
p <- p + geom_point(data = traindata, aes(x = longitude, y = latitude), color = "red")
p <- p + scale_x_continuous(limits = c(-75, -73))
p <- p + scale_y_continuous(limits = c(40, 41))
plot(p)
rm(statelines)

# Created a grid that is divided into roughly 1/2 square mile regions and assigned each
# listing to one specific categorical grid code based on its lat/long.  The idea is to attempt to capture
# a "neighborhood" or regional feature.

latlonggrid <- data.frame(traindata$listing_id, traindata$longitude, traindata$latitude)
names(latlonggrid) <- c("listing_id", "longitude", "latitude")
minlong <- min(latlonggrid$longitude, na.rm = TRUE); maxlong <- max(latlonggrid$longitude[latlonggrid$longitude < 0], na.rm = TRUE)
minlat <- min(latlonggrid$latitude[latlonggrid$latitude > 0], na.rm = TRUE); maxlat <- max(latlonggrid$latitude, na.rm = TRUE)
lat_segment <- (maxlat - minlat)/10
long_segment <- (maxlong - minlong)/10
latlonggrid$longcat <- ""
latlonggrid$latcat <- ""

catlong <- function(x) {
      as.character((maxlong - x) %/% long_segment)
}
catlat <- function(y) {
      as.character((maxlat - y) %/% lat_segment)
}
latlonggrid$longcat <- sapply(latlonggrid$longitude, catlong)
latlonggrid$latcat <- sapply(latlonggrid$latitude, catlat)
latlonggrid$longcat[latlonggrid$longcat == "-1749"] <- "unclassified"
latlonggrid$latcat[as.numeric(latlonggrid$latcat) == "1188"] <- "unclassified"
latlonggrid$gridcode <- paste(latlonggrid$longcat, latlonggrid$latcat, sep = "")
latlonggrid$gridcode <- gsub("^unclassifiedunclassified$", "unclassified", latlonggrid$gridcode)
traindata$longcat <- as.factor(latlonggrid$longcat)
traindata$latcat <- as.factor(latlonggrid$latcat)
traindata$gridcode <- as.factor(latlonggrid$gridcode)

latlonggrid <- data.frame(testdata$listing_id, testdata$longitude, testdata$latitude)
names(latlonggrid) <- c("listing_id", "longitude", "latitude")
minlong <- min(latlonggrid$longitude, na.rm = TRUE); maxlong <- max(latlonggrid$longitude[latlonggrid$longitude < 0], na.rm = TRUE)
minlat <- min(latlonggrid$latitude[latlonggrid$latitude > 0], na.rm = TRUE); maxlat <- max(latlonggrid$latitude, na.rm = TRUE)
lat_segment <- (maxlat - minlat)/10
long_segment <- (maxlong - minlong)/10
latlonggrid$longcat <- ""
latlonggrid$latcat <- ""
latlonggrid$longcat <- sapply(latlonggrid$longitude, catlong)
latlonggrid$latcat <- sapply(latlonggrid$latitude, catlat)
latlonggrid$longcat[latlonggrid$longcat == "-1740"] <- "unclassified"
latlonggrid$latcat[as.numeric(latlonggrid$latcat) == "1042"] <- "unclassified"
latlonggrid$gridcode <- paste(latlonggrid$longcat, latlonggrid$latcat, sep = "")
latlonggrid$gridcode <- gsub("^unclassifiedunclassified$", "unclassified", latlonggrid$gridcode)
testdata$longcat <- as.factor(latlonggrid$longcat)
testdata$latcat <- as.factor(latlonggrid$latcat)
testdata$gridcode <- as.factor(latlonggrid$gridcode)

testdata$gridcode <- gsub("010|06|105|11|27|83|84|86|87|97", "unclassified", testdata$gridcode)

rm(lat_segment); rm(long_segment); rm(maxlat); rm(maxlong); rm(minlat); rm(minlong)
rm(latlonggrid)

# Creating scaled per bedroom and per bathroom cost features

traindata$bedcostindex <- log((as.numeric(traindata$price)/(as.numeric(traindata$bedrooms) + 1)))
traindata$bathcostindex <- log(traindata$price/(traindata$bathrooms + 1))
traindata$bedbathindex <- log(traindata$price/(traindata$bathrooms + traindata$bedrooms + 1))
traindata$bedbathsum <- traindata$bathrooms + traindata$bedrooms

testdata$bedcostindex <- log((as.numeric(testdata$price)/(as.numeric(testdata$bedrooms) + 1)))
testdata$bathcostindex <- log(testdata$price/(testdata$bathrooms + 1))
testdata$bedbathindex <- log(testdata$price/(testdata$bathrooms + testdata$bedrooms + 1))
testdata$bedbathsum <- testdata$bathrooms + testdata$bedrooms

# Using a shapiro-test to determine whether pricing follows a normal distribution

hist(traindata$price, breaks = 10000, xlim = range(0, 50000))
sprice <- sample(traindata$price, 5000)
shapiro.test(sprice)
rm(sprice)

# The results of the shapiro test above indicate that prices are normally distributed.  
# There are numerous buildings with several listings in each one.  Calculating z-values
# for each listing that relates the price to other units in that specific building.
# For buildings with a small number of listings, this z-score is assigned to 0.

detach(package:plyr)
bb <- data.frame(traindata$listing_id, traindata$building_id, traindata$bedrooms, traindata$price)
names(bb) <- c("listing_id", "building_id", "bedrooms", "price")
bb <- group_by(bb, building_id)
bbstats <- summarize(bb, mean(price), sqrt(var(price)))
names(bbstats) <- c("building_id", "mean", "sd")
bbstats <- sqldf("SELECT bb.listing_id, bb.building_id, bb.price, bbstats.mean, bbstats.sd 
                 FROM bb, bbstats
                 WHERE bb.building_id = bbstats.building_id")
zscore <- function(obs, avg, sd) {
      z <- (obs - avg)/sd
      return(z)
}
traindata$zscorepricebuilding <- zscore(bbstats$price, bbstats$mean, bbstats$sd)
traindata$zscorepricebuilding[is.na(traindata$zscorepricebuilding)] <- 0

bb <- data.frame(testdata$listing_id, testdata$building_id, testdata$bedrooms, testdata$price)
names(bb) <- c("listing_id", "building_id", "bedrooms", "price")
bb <- group_by(bb, building_id)
bbstats <- summarize(bb, mean(price), sqrt(var(price)))
names(bbstats) <- c("building_id", "mean", "sd")
bbstats <- sqldf("SELECT bb.listing_id, bb.building_id, bb.price, bbstats.mean, bbstats.sd 
                 FROM bb, bbstats
                 WHERE bb.building_id = bbstats.building_id")

testdata$zscorepricebuilding <- zscore(bbstats$price, bbstats$mean, bbstats$sd)
testdata$zscorepricebuilding[is.na(testdata$zscorepricebuilding)] <- 0

rm(bb); rm(bbstats)

high <- function(id) {
      length(which(traindata$interest_level == "high" & traindata$building_id == id))/length(which(traindata$building_id == id))
}
medium <- function(id) {
      length(which(traindata$interest_level == "medium" & traindata$building_id == id))/length(which(traindata$building_id == id))
}
low <- function(id) {
      length(which(traindata$interest_level == "low" & traindata$building_id == id))/length(which(traindata$building_id == id))
}

manno <- function(id) {
      log(length(which(traindata$building_id == id)))
}

bid <- as.data.frame(matrix(data = NA, nrow = length(unique(traindata$building_id)), ncol = 5))
names(bid) <- c("id", "high", "med", "low", "conf")
bid$id <- unique(traindata$building_id)
bid$high <- sapply(bid$id, high)
bid$med <- sapply(bid$id, medium)
bid$low <- sapply(bid$id, low)
bid$conf <- sapply(bid$id, manno)

traindata <- merge(traindata,
                   bid,
                   by.x = "building_id",
                   by.y = "id",
                   all.x = TRUE)

testdata <- merge(testdata,
                  bid,
                  by.x = "building_id",
                  by.y = "id",
                  all.x = TRUE)

testdata$low[is.na(testdata$low)] <- mean(traindata$low)
testdata$med[is.na(testdata$med)] <- mean(traindata$med)
testdata$high[is.na(testdata$high)] <- mean(traindata$high)
testdata$conf[is.na(testdata$conf)] <- min(traindata$conf)

traindata <- select(traindata, -building_id)
testdata <- select(testdata, -building_id)

library(plyr)

# Calculating various listing densities.  Creating features that determine local 
# densities of similar listing types based on number of bedrooms and bathrooms.

a1 <- sapply(1:nrow(traindata), 
       function(i) length(which(traindata$latitude <= traindata$latitude[i] + 0.005 & 
                                  traindata$latitude >= traindata$latitude[i] - 0.005 & 
                                  traindata$longitude <= traindata$longitude[i] + 0.005 & 
                                  traindata$longitude >= traindata$longitude[i] - 0.005 & 
                                  traindata$bathrooms <= traindata$bathrooms[i] + 1 & 
                                  traindata$bathrooms >= traindata$bathrooms[i] - 1 & 
                                  traindata$bedrooms <= traindata$bedrooms[i] + 1 & 
                                  traindata$bedrooms >= traindata$bedrooms[i] - 1)))

a2 <- sapply(1:nrow(traindata), 
             function(i) length(which(traindata$latitude <= traindata$latitude[i] + 0.01 & 
                                      traindata$latitude >= traindata$latitude[i] - 0.01 & 
                                      traindata$longitude <= traindata$longitude[i] + 0.01 & 
                                      traindata$longitude >= traindata$longitude[i] - 0.01 & 
                                      traindata$bathrooms <= traindata$bathrooms[i] + 1 & 
                                      traindata$bathrooms >= traindata$bathrooms[i] - 1 & 
                                      traindata$bedrooms <= traindata$bedrooms[i] + 1 & 
                                      traindata$bedrooms >= traindata$bedrooms[i] - 1)))

a3 <- sapply(1:nrow(traindata), 
             function(i) length(which(traindata$latitude <= traindata$latitude[i] + 0.015 & 
                                      traindata$latitude >= traindata$latitude[i] - 0.015 & 
                                      traindata$longitude <= traindata$longitude[i] + 0.015 & 
                                      traindata$longitude >= traindata$longitude[i] - 0.015 & 
                                      traindata$bathrooms <= traindata$bathrooms[i] + 1 & 
                                      traindata$bathrooms >= traindata$bathrooms[i] - 1 & 
                                      traindata$bedrooms <= traindata$bedrooms[i] + 1 & 
                                      traindata$bedrooms >= traindata$bedrooms[i] - 1)))

traindata$localdensity <- a1/a2
traindata$lesslocaldensity <- a1/a3
rm(a1); rm(a2); rm(a3)

traindata$localdensity[traindata$localdensity == "NaN"] <- 0

a1 <- sapply(1:nrow(testdata), 
             function(i) length(which(testdata$latitude <= testdata$latitude[i] + 0.005 & 
                                            testdata$latitude >= testdata$latitude[i] - 0.005 & 
                                            testdata$longitude <= testdata$longitude[i] + 0.005 & 
                                            testdata$longitude >= testdata$longitude[i] - 0.005 & 
                                            testdata$bathrooms <= testdata$bathrooms[i] + 1 & 
                                            testdata$bathrooms >= testdata$bathrooms[i] - 1 & 
                                            testdata$bedrooms <= testdata$bedrooms[i] + 1 & 
                                            testdata$bedrooms >= testdata$bedrooms[i] - 1)))

a2 <- sapply(1:nrow(testdata), 
             function(i) length(which(testdata$latitude <= testdata$latitude[i] + 0.01 & 
                                            testdata$latitude >= testdata$latitude[i] - 0.01 & 
                                            testdata$longitude <= testdata$longitude[i] + 0.01 & 
                                            testdata$longitude >= testdata$longitude[i] - 0.01 & 
                                            testdata$bathrooms <= testdata$bathrooms[i] + 1 & 
                                            testdata$bathrooms >= testdata$bathrooms[i] - 1 & 
                                            testdata$bedrooms <= testdata$bedrooms[i] + 1 & 
                                            testdata$bedrooms >= testdata$bedrooms[i] - 1)))

a3 <- sapply(1:nrow(testdata), 
             function(i) length(which(testdata$latitude <= testdata$latitude[i] + 0.015 & 
                                            testdata$latitude >= testdata$latitude[i] - 0.015 & 
                                            testdata$longitude <= testdata$longitude[i] + 0.015 & 
                                            testdata$longitude >= testdata$longitude[i] - 0.015 & 
                                            testdata$bathrooms <= testdata$bathrooms[i] + 1 & 
                                            testdata$bathrooms >= testdata$bathrooms[i] - 1 & 
                                            testdata$bedrooms <= testdata$bedrooms[i] + 1 & 
                                            testdata$bedrooms >= testdata$bedrooms[i] - 1)))

testdata$localdensity <- a1/a2
testdata$lesslocaldensity <- a1/a3
rm(a1); rm(a2); rm(a3)

testdata$localdensity[testdata$localdensity == "NaN"] <- 0

# Generating features that attempt to isolate the influence of individual managers on rental interest.

high <- function(id) {
      length(which(traindata$interest_level == "high" & traindata$manager_id == id))/length(which(traindata$manager_id == id))
}
medium <- function(id) {
      length(which(traindata$interest_level == "medium" & traindata$manager_id == id))/length(which(traindata$manager_id == id))
}
low <- function(id) {
      length(which(traindata$interest_level == "low" & traindata$manager_id == id))/length(which(traindata$manager_id == id))
}

manno <- function(id) {
      log(length(which(traindata$manager_id == id)))
}

manager <- as.data.frame(matrix(data = NA, nrow = length(unique(traindata$manager_id)), ncol = 5))
names(manager) <- c("id", "high", "med", "low", "conf")
manager$id <- unique(traindata$manager_id)
manager$high <- sapply(manager$id, high)
manager$med <- sapply(manager$id, medium)
manager$low <- sapply(manager$id, low)
manager$conf <- sapply(manager$id, manno)

traindata <- merge(traindata,
                   manager,
                   by.x = "manager_id",
                   by.y = "id",
                   all.x = TRUE)

traindata <- select(traindata, -manager_id)

testdata <- merge(testdata,
                  manager,
                  by.x = "manager_id",
                  by.y = "id",
                  all.x = TRUE)
testdata <- select(testdata, -manager_id)

testdata$low.y[is.na(testdata$low.y)] <- mean(traindata$low.y)
testdata$med.y[is.na(testdata$med.y)] <- mean(traindata$med.y)
testdata$high.y[is.na(testdata$high.y)] <- mean(traindata$high.y)
testdata$conf.y[is.na(testdata$conf.y)] <- min(traindata$conf.y)

rm(manager)

# Initial analysis of description field.  Calculating the character length and the relative
# frequency of capital letters in each description.

traindata$lengthdesc <- sapply(traindata$description, nchar)
traindata$lengthdesc <- as.numeric(traindata$lengthdesc)
traindata$perccapdesc <- parSapply(cl, 
                                   regmatches(
                                         traindata$description, 
                                         gregexpr("[A-Z]", 
                                                  traindata$description, 
                                                  perl=TRUE)), 
                                   length)
traindata$perccapdesc <- (as.numeric(traindata$perccapdesc) + 1)/(as.numeric(traindata$lengthdesc) + 1)
traindata$perex <- parSapply(cl, regmatches(
                                          traindata$description,
                                          gregexpr("!",
                                                   traindata$description,
                                                   perl = TRUE)),
                                    length)

testdata$lengthdesc <- sapply(testdata$description, nchar)
testdata$lengthdesc <- as.numeric(testdata$lengthdesc)
testdata$perccapdesc <- parSapply(cl, 
                                  regmatches(
                                        testdata$description, 
                                        gregexpr("[A-Z]", 
                                                 testdata$description, 
                                                 perl=TRUE)), 
                                  length)
testdata$perccapdesc <- (as.numeric(testdata$perccapdesc) + 1)/(as.numeric(testdata$lengthdesc) + 1)
testdata$perex <- parSapply(cl, regmatches(
                                          testdata$description,
                                          gregexpr("!",
                                              testdata$description,
                                              perl = TRUE)),
                                          length)


# Sentiment analysis of listing description.  Each description is analyzed and quantified 
# based on 8 different emotional sentiments.   

traindata$description <- as.character(traindata$description)
sentiment <- as.data.frame(syuzhet::get_sentiment(traindata$description), 
                           method = "stanford")
names(sentiment) <- c("sentimentrating")
sentiment$sentimentrating <- scale(sentiment$sentimentrating)
traindata$sentimentrating <- sentiment$sentimentrating
rm(sentiment)
sentiment <- get_nrc_sentiment(traindata$description)
rowsumsentiment <- rowSums(sentiment) + 1
for (i in 1:nrow(sentiment)) {
      sentiment[i,] <- sentiment[i,]/rowsumsentiment[i]
}
sentiment <- as.data.frame(apply(sentiment, 2, scale))
traindata <- data.frame(traindata, sentiment)

testdata$description <- as.character(testdata$description)
sentiment <- as.data.frame(syuzhet::get_sentiment(testdata$description), 
                           method = "stanford")
names(sentiment) <- c("sentimentrating")
sentiment$sentimentrating <- scale(sentiment$sentimentrating)
testdata$sentimentrating <- sentiment$sentimentrating
rm(sentiment)
sentiment <- get_nrc_sentiment(testdata$description)
rowsumsentiment <- rowSums(sentiment) + 1
for (i in 1:nrow(sentiment)) {
      sentiment[i,] <- sentiment[i,]/rowsumsentiment[i]
}
sentiment <- as.data.frame(apply(sentiment, 2, scale))
testdata <- data.frame(testdata, sentiment)

rm(sentiment); rm(rowsumsentiment)

traindata <- select(traindata, -description)
testdata <- select(testdata, -description)

# Analysis of photo names.  Stock imagery is commonly used and can sometimes be
# distinguished from images taken on a per-listing basis. The results of the code 
# below indicate that the image names are all unique and therefore determining 
# if/when stock photography was used is not possible without further image analysis
# that will not be performed here due to the collective size of the image files.  

# It should be noted that the image names were found by other Kaggler's to have a profound
# effect on listing interest because of an inherent time feature built into the name.  

photos <- traindata[,c(9,12)]
names(photos) <- c("link", "num")
photos <- separate(photos,
                   link,
                   into = paste("photonum", 1:max(photos$num), sep = ""),
                   sep = ",")
photos$id <- 1:nrow(photos)
photos <- select(photos, -num)
photos <- gather(data = photos,
                 key = id,
                 value = all,
                 photonum1:photonum68)
photos <- photos[,-2]
photos <- photos[which(photos$all != ""),]
photos <- separate(data = photos,
                   col = all,
                   into = c("listingcode", "photocode"),
                   sep = "_")
length(unique(photos$photocode))

rm(photos)

traindata <- select(traindata, -photos)
testdata <- select(testdata, -photos)

# Splitting the creation time of the listings to create categorical variables based 
# on month, day, and time of day that each listing was created.

created <- as.data.frame(traindata$created)
names(created) <- c("datetime")
created$datetime <- as.character(created$datetime)
created <- separate(created,
                    col = datetime,
                    into = c("datecreated", "timecreated"),
                    sep = " ")
created$datecreated <- as.Date(created$datecreated)
created$daycreated <- wday(created$datecreated, label=TRUE)
created$month <- month(created$datecreated, label = TRUE)
created$timecreated <- as.numeric(substring(created$timecreated, 1, 2))
created$timeofday <- case_when (
      created$timecreated < 6 ~ "overnight",
      created$timecreated < 12 & created$timecreated >= 6 ~ "morning",
      created$timecreated < 18 & created$timecreated >= 12 ~ "afternoon",
      created$timecreated < 24 & created$timecreated >= 18 ~ "evening"
)
created$dayspassed <- created$datecreated - min(created$datecreated)
created$datecreated <- format(as.Date(created$datecreated, format = "%Y-%m-%d"), "%d")
created$daycreated <- as.factor(as.character(as.factor(created$daycreated)))
created$month <- as.factor(as.character(as.factor(created$month)))
created$timeofday <- as.factor(created$timeofday)
created <- created[,-c(1:2)]
traindata <- data.frame(traindata, created)

created <- as.data.frame(testdata$created)
names(created) <- c("datetime")
created$datetime <- as.character(created$datetime)
created <- separate(created,
                    col = datetime,
                    into = c("datecreated", "timecreated"),
                    sep = " ")
created$datecreated <- as.Date(created$datecreated)
created$daycreated <- wday(created$datecreated, label=TRUE)
created$month <- month(created$datecreated, label = TRUE)
created$timecreated <- as.numeric(substring(created$timecreated, 1, 2))
created$timeofday <- case_when (
      created$timecreated < 6 ~ "overnight",
      created$timecreated < 12 & created$timecreated >= 6 ~ "morning",
      created$timecreated < 18 & created$timecreated >= 12 ~ "afternoon",
      created$timecreated < 24 & created$timecreated >= 18 ~ "evening"
)
created$dayspassed <- created$datecreated - min(created$datecreated)
created$datecreated <- format(as.Date(created$datecreated, format = "%Y-%m-%d"), "%d")
created$daycreated <- as.factor(as.character(as.factor(created$daycreated)))
created$month <- as.factor(as.character(as.factor(created$month)))
created$timeofday <- as.factor(created$timeofday)
created <- created[,-c(1:2)]
testdata <- data.frame(testdata, created)

rm(created)

traindata <- select(traindata, -created)
testdata <- select(testdata, -created)

# Street and Display Address information and data normalization.  Because of the lack in uniformity in which
# addresses were entered, creating factor variables from each different address would be a computationally 
# difficult task--well beyond the means of my home computer--and likely would prove to be useless.  
# Normalizing the address fields would make that much more feasible.  Example:  "256 fifth st", "256 5th ST"
# and "256 5th str" all represent the same address but would be interpreted as 3 distinct listings before
# normalizing.

traindata$street_address <- tolower(traindata$street_address)
traindata$display_address <- tolower(traindata$display_address)
traindata$street_address <- trimws(traindata$street_address, which = "both")
traindata$display_address <- trimws(traindata$display_address, which = "both")
traindata$street_address <- gsub(" n |^n ", " north ", traindata$street_address)
traindata$street_address <- gsub(" s |^s ", " south ", traindata$street_address)
traindata$street_address <- gsub(" e |^e ", " east ", traindata$street_address)
traindata$street_address <- gsub(" w |^w ", " west ", traindata$street_address)
traindata$display_address <- gsub(" n |^n ", " north ", traindata$display_address)
traindata$display_address <- gsub(" s |^s ", " south ", traindata$display_address)
traindata$display_address <- gsub(" e |^e ", " east ", traindata$display_address)
traindata$display_address <- gsub(" w |^w ", " west ", traindata$display_address)
traindata$street_address <- trimws(traindata$street_address, which = "both")
traindata$display_address <- trimws(traindata$display_address, which = "both")
traindata$street_address <- gsub("[[:punct:]]", "", traindata$street_address)
traindata$display_address <- gsub("[[:punct:]]", "", traindata$display_address)
traindata$street_address <- gsub(" st$", " street", traindata$street_address)
traindata$street_address <- gsub(" ave$", " avenue", traindata$street_address)
traindata$street_address <- gsub(" pl$", " place", traindata$street_address)
traindata$display_address <- gsub(" st$", " street", traindata$display_address)
traindata$display_address <- gsub(" ave$", " avenue", traindata$display_address)
traindata$display_address <- gsub(" pl$", " place", traindata$display_address)
traindata$street_address <- gsub("1st|first", "1", traindata$street_address)
traindata$street_address <- gsub("2nd|second", "2", traindata$street_address)
traindata$street_address <- gsub("3rd|third", "3", traindata$street_address)
traindata$street_address <- gsub("4th|fourth", "4", traindata$street_address)
traindata$street_address <- gsub("5th|fifth", "5", traindata$street_address)
traindata$street_address <- gsub("6th|sixth", "6", traindata$street_address)
traindata$street_address <- gsub("7th|seventh", "7", traindata$street_address)
traindata$street_address <- gsub("8th|eighth", "8", traindata$street_address)
traindata$street_address <- gsub("9th|ninth", "9", traindata$street_address)
traindata$street_address <- gsub("10th|tenth|ten", "10", traindata$street_address)
traindata$street_address <- gsub("11th|eleventh|eleven", "11", traindata$street_address)
traindata$street_address <- gsub("12th|twelfth|twelve", "12", traindata$street_address)
traindata$street_address <- gsub("13th|thirteenth|thirteen", "13", traindata$street_address)
traindata$street_address <- gsub("0th", "", traindata$street_address)
traindata$street_address <- gsub("  ", " ", traindata$street_address)

traindata$display_address <- gsub("1st|first", "1", traindata$display_address)
traindata$display_address <- gsub("2nd|second", "2", traindata$display_address)
traindata$display_address <- gsub("3rd|third", "3", traindata$display_address)
traindata$display_address <- gsub("4th|fourth", "4", traindata$display_address)
traindata$display_address <- gsub("5th|fifth", "5", traindata$display_address)
traindata$display_address <- gsub("6th|sixth", "6", traindata$display_address)
traindata$display_address <- gsub("7th|seventh", "7", traindata$display_address)
traindata$display_address <- gsub("8th|eighth", "8", traindata$display_address)
traindata$display_address <- gsub("9th|ninth", "9", traindata$display_address)
traindata$display_address <- gsub("10th|tenth|ten", "10", traindata$display_address)
traindata$display_address <- gsub("11th|eleventh|eleven", "11", traindata$display_address)
traindata$display_address <- gsub("12th|twelfth|twelve", "12", traindata$display_address)
traindata$display_address <- gsub("13th|thirteenth|thirteen", "13", traindata$display_address)
traindata$display_address <- gsub("0th", "", traindata$display_address)
traindata$display_address <- gsub("  ", " ", traindata$display_address)

traindata$display_address[which(traindata$display_address == traindata$street_address)] <- 
      str_split_fixed(traindata$street_address[which(traindata$display_address == traindata$street_address)], " ", n = 2)[,2]
traindata$display_address[which(traindata$display_address != str_split_fixed(traindata$street_address, " ", n = 2)[,2])] <-
      str_split_fixed(traindata$street_address[which(traindata$display_address != str_split_fixed(traindata$street_address, " ", n = 2)[,2])], " ", n = 2)[,2]
traindata <- select(traindata, -street_address)


testdata$street_address <- tolower(testdata$street_address)
testdata$display_address <- tolower(testdata$display_address)
testdata$street_address <- trimws(testdata$street_address, which = "both")
testdata$display_address <- trimws(testdata$display_address, which = "both")
testdata$street_address <- gsub(" n |^n ", " north ", testdata$street_address)
testdata$street_address <- gsub(" s |^s ", " south ", testdata$street_address)
testdata$street_address <- gsub(" e |^e ", " east ", testdata$street_address)
testdata$street_address <- gsub(" w |^w ", " west ", testdata$street_address)
testdata$display_address <- gsub(" n |^n ", " north ", testdata$display_address)
testdata$display_address <- gsub(" s |^s ", " south ", testdata$display_address)
testdata$display_address <- gsub(" e |^e ", " east ", testdata$display_address)
testdata$display_address <- gsub(" w |^w ", " west ", testdata$display_address)
testdata$street_address <- trimws(testdata$street_address, which = "both")
testdata$display_address <- trimws(testdata$display_address, which = "both")
testdata$street_address <- gsub("[[:punct:]]", "", testdata$street_address)
testdata$display_address <- gsub("[[:punct:]]", "", testdata$display_address)
testdata$street_address <- gsub(" st$", " street", testdata$street_address)
testdata$street_address <- gsub(" ave$", " avenue", testdata$street_address)
testdata$street_address <- gsub(" pl$", " place", testdata$street_address)
testdata$display_address <- gsub(" st$", " street", testdata$display_address)
testdata$display_address <- gsub(" ave$", " avenue", testdata$display_address)
testdata$display_address <- gsub(" pl$", " place", testdata$display_address)
testdata$street_address <- gsub("1st|first", "1", testdata$street_address)
testdata$street_address <- gsub("2nd|second", "2", testdata$street_address)
testdata$street_address <- gsub("3rd|third", "3", testdata$street_address)
testdata$street_address <- gsub("4th|fourth", "4", testdata$street_address)
testdata$street_address <- gsub("5th|fifth", "5", testdata$street_address)
testdata$street_address <- gsub("6th|sixth", "6", testdata$street_address)
testdata$street_address <- gsub("7th|seventh", "7", testdata$street_address)
testdata$street_address <- gsub("8th|eighth", "8", testdata$street_address)
testdata$street_address <- gsub("9th|ninth", "9", testdata$street_address)
testdata$street_address <- gsub("10th|tenth|ten", "10", testdata$street_address)
testdata$street_address <- gsub("11th|eleventh|eleven", "11", testdata$street_address)
testdata$street_address <- gsub("12th|twelfth|twelve", "12", testdata$street_address)
testdata$street_address <- gsub("13th|thirteenth|thirteen", "13", testdata$street_address)
testdata$street_address <- gsub("0th", "", testdata$street_address)
testdata$street_address <- gsub("  ", " ", testdata$street_address)

testdata$display_address <- gsub("1st|first", "1", testdata$display_address)
testdata$display_address <- gsub("2nd|second", "2", testdata$display_address)
testdata$display_address <- gsub("3rd|third", "3", testdata$display_address)
testdata$display_address <- gsub("4th|fourth", "4", testdata$display_address)
testdata$display_address <- gsub("5th|fifth", "5", testdata$display_address)
testdata$display_address <- gsub("6th|sixth", "6", testdata$display_address)
testdata$display_address <- gsub("7th|seventh", "7", testdata$display_address)
testdata$display_address <- gsub("8th|eighth", "8", testdata$display_address)
testdata$display_address <- gsub("9th|ninth", "9", testdata$display_address)
testdata$display_address <- gsub("10th|tenth|ten", "10", testdata$display_address)
testdata$display_address <- gsub("11th|eleventh|eleven", "11", testdata$display_address)
testdata$display_address <- gsub("12th|twelfth|twelve", "12", testdata$display_address)
testdata$display_address <- gsub("13th|thirteenth|thirteen", "13", testdata$display_address)
testdata$display_address <- gsub("0th", "", testdata$display_address)
testdata$display_address <- gsub("  ", " ", testdata$display_address)

testdata$display_address[which(testdata$display_address == testdata$street_address)] <- 
      str_split_fixed(testdata$street_address[which(testdata$display_address == testdata$street_address)], " ", n = 2)[,2]
testdata$display_address[which(testdata$display_address != str_split_fixed(testdata$street_address, " ", n = 2)[,2])] <-
      str_split_fixed(testdata$street_address[which(testdata$display_address != str_split_fixed(testdata$street_address, " ", n = 2)[,2])], " ", n = 2)[,2]
testdata <- select(testdata, -street_address)

road <- function(address) {
      nw <- str_count(address, pattern = " ") + 1
      str_split_fixed(address, pattern = " ", n = nw)[nw]
}

traindata$roadtype <- sapply(traindata$display_address, road)
traindata$roadtype <- trimws(traindata$roadtype, which = "both")
traindata$roadtype <- gsub("blvd", "boulevard", traindata$roadtype)
traindata$roadtype <- gsub("pkwy", "parkway", traindata$roadtype)
traindata$roadtype <- gsub("^rd$", "road", traindata$roadtype)
traindata$roadtype <- gsub("^dr$", "drive", traindata$roadtype)
rt <- c("street", "avenue", "place", "boulevard", "parkway", "court", "square", "road", "drive", "lane", "terrace", "broadway", "north", "east", "south", "west", "plaza", "district")
for (i in 1:nrow(traindata)) {
      traindata$roadtype[i] <- ifelse(traindata$roadtype[i] %in% rt, traindata$roadtype[i], "unclassified")
}
traindata$roadtype <- as.factor(traindata$roadtype)

testdata$roadtype <- sapply(testdata$display_address, road)
testdata$roadtype <- trimws(testdata$roadtype, which = "both")
testdata$roadtype <- gsub("blvd", "boulevard", testdata$roadtype)
testdata$roadtype <- gsub("pkwy", "parkway", testdata$roadtype)
testdata$roadtype <- gsub("^rd$", "road", testdata$roadtype)
testdata$roadtype <- gsub("^dr$", "drive", testdata$roadtype)
rt <- c("street", "avenue", "place", "boulevard", "parkway", "court", "square", "road", "drive", "lane", "terrace", "broadway", "north", "east", "south", "west", "plaza", "district")
for (i in 1:nrow(testdata)) {
      testdata$roadtype[i] <- ifelse(testdata$roadtype[i] %in% rt, testdata$roadtype[i], "unclassified")
}
testdata$roadtype <- as.factor(testdata$roadtype)

# Creating z-score features that relate street address to listing interest.


high <- function(id) {
      length(which(traindata$interest_level == "high" & traindata$display_address == id))/length(which(traindata$display_address == id))
}
medium <- function(id) {
      length(which(traindata$interest_level == "medium" & traindata$display_address == id))/length(which(traindata$display_address == id))
}
low <- function(id) {
      length(which(traindata$interest_level == "low" & traindata$display_address == id))/length(which(traindata$display_address == id))
}

manno <- function(id) {
      log(length(which(traindata$display_address == id)))
}

da <- as.data.frame(matrix(data = NA, nrow = length(unique(traindata$display_address)), ncol = 5))
names(da) <- c("id", "high", "med", "low", "conf")
da$id <- unique(traindata$display_address)
da$high <- sapply(da$id, high)
da$med <- sapply(da$id, medium)
da$low <- sapply(da$id, low)
da$conf <- sapply(da$id, manno)

traindata <- merge(traindata,
                   da,
                   by.x = "display_address",
                   by.y = "id",
                   all.x = TRUE)

testdata <- merge(testdata,
                  da,
                  by.x = "display_address",
                  by.y = "id",
                  all.x = TRUE)

testdata$low[is.na(testdata$low)] <- mean(traindata$low)
testdata$med[is.na(testdata$med)] <- mean(traindata$med)
testdata$high[is.na(testdata$high)] <- mean(traindata$high)
testdata$conf[is.na(testdata$conf)] <- min(traindata$conf)


traindata <- select(traindata ,-display_address)
testdata <- select(testdata ,-display_address)

rm(bid);rm(da)

# save.image("C:/Users/Bryan/Google Drive/Kaggle/TwoSigma/preprediction.RData")

# Rearranging fields so that relevant fields are adjacent; a little bit of data science OCD.

traindata <- traindata[,c(7,1:6,8:131)]
testdata <- testdata[,c(7,1:6,8:131)]

# Identifying factor and numerical features and setting them as such

factorvars <- c(7,127,128,129,130,131)
numvars <- setdiff(c(1:ncol(traindata)), factorvars)

listing_id_test <- testdata$listing_id
traindata$listing_id <- scale(traindata$listing_id)
testdata$listing_id <- scale(testdata$listing_id)

for (i in factorvars) {
      traindata[,i] <- as.factor(traindata[,i])
}

for (i in numvars) {
      traindata[,i] <- as.numeric(traindata[,i])
}

factorvars <- c(127,128,129,130,131)
numvars <- setdiff(c(1:ncol(traindata)), factorvars)

for (i in factorvars) {
      testdata[,i] <- as.factor(testdata[,i])
}

for (i in numvars) {
      testdata[,i] <- as.numeric(testdata[,i])
}

# Identifying highly correlated features and removing them

corrtrain <- cor(traindata[,numvars])
correlated <- findCorrelation(corrtrain, cutoff = 0.95)
traindata <- traindata[,-correlated]
testdata <- testdata[,-correlated]

rm(corrtrain); rm(correlated); rm(factorvars); rm(i); rm(j); rm(numvars); rm(rt)

# Creating training model.  After several modeling attempts, I settled on a boosted tree model.  

set.seed(314)

intrain <- createDataPartition(traindata$interest_level, p = 0.7, list = FALSE)
train_set <- traindata[intrain,]
val_set <- traindata[-intrain,]

xgbtree_grid <- expand.grid(nrounds = c(145, 150, 155),
                            max_depth = 100,
                            eta = 0.30,
                            gamma = 0,
                            colsample_bytree = c(0.55, 0.6, 0.65),
                            min_child_weight = 1,
                            subsample = c(0.9, 1))


tc <- trainControl(method = "cv", 
                   number = 10, 
                   repeats = 5,
                   classProbs = TRUE)

xgbtree_model <- train(interest_level ~.,
                  data = train_set,
                  method = "xgbTree",
                  trControl = tc,
                  tuneGrid = xgbtree_grid,
                  objective = "multi:softprob",
                  eval_metric = "mlogloss")

# Generating the testdata predictions in the format that Kaggle requires

prediction_xgb <- predict(xgbtree_model, testdata, type = "prob")
confusionMatrix(prediction_xgb, val_set$interest_level)

xgb_prediction <- cbind(listing_id_test, prediction_xgb)
xgb_prediction <- xgb_prediction[,c(1,2,4,3)]
names(xgb_prediction) <- c("listing_id", "high", "medium", "low")
xgb_prediction[,c(2:4)] <- round(xgb_prediction[,c(2:4)], 6)

# And writing the prediction file to disk for upload

write.csv(xgb_prediction, "C:/Users/Bryan/Google Drive/Kaggle/TwoSigma/xgb.csv", row.names = FALSE)
save.image("C:/Users/Bryan/Google Drive/Kaggle/TwoSigma/prediction2.RData")


