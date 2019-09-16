# libraries --------------
install.packages("readr")
install.packages("caret")
install.packages("DataExplorer")
install.packages("mlbench")

pacman::p_load()
library(readr)
library(caret)
library(ggplot2)
library(reshape2)
library(DataExplorer)
library(MASS)
library(tidyr)
library(mlbench)


# Reading data -------------

complete_data <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_2/CompleteResponses.csv")
missing_data <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_2/SurveyIncomplete.csv")
missing_data1 <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_2/SurveyIncomplete1.csv")
summary(complete_data)
summary(complete_data)
str(complete_data)
names(complete_data)
is.na(complete_data)
attach(complete_data)
detach(complete_data)


# change variables data type -------------
## use factor insteadd, and add levels ex. 

complete_data$elevel <- as.factor(complete_data$elevel) 
complete_data$car <- as.factor(complete_data$car)
complete_data$zipcode <- as.factor(complete_data$zipcode)
complete_data$brand <- as.factor(complete_data$brand)
str(complete_data)

missing_data$elevel <- as.factor(missing_data$elevel) 
missing_data$car <- as.factor(missing_data$car)
missing_data$zipcode <- as.factor(missing_data$zipcode)
missing_data$brand <- as.factor(missing_data$brand)
str(missing_data)

missing_data1$elevel <- as.factor(missing_data1$elevel) 
missing_data1$car <- as.factor(missing_data1$car)
missing_data1$zipcode <- as.factor(missing_data1$zipcode)
missing_data1$brand <- as.factor(missing_data1$brand)
str(missing_data1)

# Data exploration ------------------
### gggally
create_report(complete_data)
create_report(missing_data1)
melt.data <- melt(complete_data)
head(melt.data)
 ggplot(data = melt.data, aes(x = value)) + 
  stat_density() + 
  facet_wrap(~variable, scales = "free")
boxplot(complete_data$salary) 
#### boxplot(data$cal1)$out list of outliers
boxplot(complete_data$credit)
test.plot <- ggplot(complete_data, aes(x=age, y=salary, color=brand)) +
  geom_point() + geom_smooth() +
  ggtitle("Customer brand preference")

test1.plot <- ggplot(complete_data, aes(x=age, y=credit, color=brand)) +
  geom_point() +
  ggtitle("Customer brand preference")

test2.plot <- ggplot(complete_data, aes(x=salary, y=credit, color=brand)) +
  geom_point() +
  ggtitle("Customer brand preference")

test3.plot <- ggplot(complete_data, aes(x=elevel,fill=brand)) +
  geom_bar() +
  ggtitle("Customer brand preference")


# making a subset of the data ------------
brand_0 <- subset(complete_data, brand == 0)
brand_1 <- subset(complete_data, brand == 1)

ggplot(brand_0, aes(x=age, y=salary)) + geom_point()
ggplot(brand_1, aes(x=age, y=salary)) + geom_point()



# Finding which transformation is the best -----------
a <- boxcox(salary ~ credit)
lambda <- a$x
lik <- a$y 
bc <- cbind(lambda, lik)
sorted_bc <- bc[order(-lik),]
head(sorted_bc, n = 10)

# Transform data -----------
complete_data$trans_salary <- sqrt(salary)
complete_data$trans_age <- sqrt(age)
complete_data$trans_credit <- sqrt(credit)

# Partition data --------------
set.seed(12345)
Intrain <- createDataPartition(complete_data$brand, p = .75, list = F)
training <- complete_data[Intrain,]
testing <- complete_data[-Intrain,]
nrow(training)
nrow(testing)

# Parallel working -------------
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
stopCluster(cl)

# C5.0 model --------------
ctrl <- trainControl(method = "repeatedcv", repeats = 1)

C50_Fit <- train(brand ~ .,
                data = training,
                method = "C5.0",
                tuneLength = 2,
                trControl = ctrl,
                preProc = c("center", "scale"))

# RF model ------------------
rfGrid = data.frame(mtry = c(1,2,3,4,5))

rf_Fit <- train(brand ~ .,
                 data = training,
                 method = "rf",
                 tuneGrid = rfGrid,
                 trControl = ctrl,
                 preProc = c("center", "scale"))

# Checking the weight of the variables in the model ------
varImp(C50_Fit)
varImp(rf_Fit)
predictors(C50_Fit) # checks which variables were used at the end

# compare models -------------
resamps <- resamples(list(C5.0 = C50_Fit, rf = rf_Fit))
summary(resamps)
xyplot(resamps, what = "BlandAltman")


# predicting brand in the test data ------------
C50_brand <- predict(C50_Fit, newdata = testing)
str(C50_brand)
C50_props <- predict(C50_Fit, newdata = testing, type = "prob")
head(C50_props)
summary(C50_brand)
postResample(C50_brand, testing$brand)

cm <- confusionMatrix(C50_brand, testing$brand)

# predicting brand in the incomplete data ------------
C50_missing <- predict(C50_Fit, newdata = missing_data)
str(C50_missing)
C50_missing_props <- predict(C50_Fit, newdata = missing_data, type = "prob")
head(C50_missing_props)
summary(C50_missing)


# plotting --------------
plot(brand)
plot(C50_brand)

results <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_2/results.csv")
summary(results)
str(results)

plot(results$number~results$Brand)

plot1 <- ggplot(results, aes(x=Brand, y=number, label= number)) +
  geom_bar(stat="identity", aes(fill=forcats::fct_rev(type))) +
  xlab("Brand") + ylab("Number of customers") +
  ggtitle("Customer brand preference") +
  labs(fill="Data type") + geom_text(size = 5, position = position_stack(vjust = 0.5))


plot2 <- plot1 + plot.settings

# General plot settings ----------------
plot.settings <- theme(
  axis.line.x =       element_line(colour = "black", size = 1),                                                       # Settings x-axis line
  axis.line.y =       element_line(colour = "black", size = 1),                                                       # Settings y-axis line 
  axis.text.x =       element_text(colour = "black", size = 16, lineheight = 0.9, vjust = 1, face = "bold"),        # Font x-axis 
  axis.text.y =       element_text(colour = "black", size = 16, lineheight = 0.9, hjust = 1),                         # Font y-axis
  axis.ticks =        element_line(colour = "black", size = 0.3),                                                     # Color/thickness axis ticks
  axis.title.x =      element_text(size = 20, vjust = 1, face = "bold", margin = margin(10,1,1,1)),                   # Font x-axis title
  axis.title.y =      element_text(size = 20, angle = 90, vjust = 1, face = "bold", margin = margin(1,10,1,1)),       # Font y-axis title
  
  legend.background = element_rect(colour=NA),                                                                        # Background color legend
  legend.key =        element_blank(),                                                                                # Background color legend key
  legend.key.size =   unit(1.2, "lines"),                                                                             # Size legend key
  legend.text =       element_text(size = 18),                                                                        # Font legend text
  legend.title =      element_text(size = 20, face = "bold", hjust = 0),                                              # Font legend title  
  legend.position =   "right",                                                                                        # Legend position
  
  panel.background =  element_blank(),                                                                                # Background color graph
  panel.border =      element_blank(),                                                                                # Border around graph (use element_rect())
  panel.grid.major =  element_blank(),                                                                                # Major gridlines (use element_line())
  panel.grid.minor =  element_blank(),                                                                                # Minor gridlines (use element_line())
  panel.margin =      unit(1, "lines"),                                                                               # Panel margins
  
  strip.background =  element_rect(fill = "grey80", colour = "grey50"),                                               # Background colour strip 
  strip.text.x =      element_text(size = 20),                                                                        # Font strip text x-axis
  strip.text.y =      element_text(size = 20, angle = -90),                                                           # Font strip text y-axis
  
  plot.background =   element_rect(colour = NA),                                                                      # Background color of entire plot
  plot.title =        element_text(size = 20, face = "bold", hjust = 0.5),                                                                        # Font plot title 
  plot.margin =       unit(c(1, 1, 1, 1), "lines")                                                                    # Plot margins
)
