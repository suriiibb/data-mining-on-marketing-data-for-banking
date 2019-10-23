bank_data<-read.csv("cleaned.csv",head=TRUE)

# plot bar plot categorical variables
par(mfrow=c(2,5))
y.frequency = table(bank_data$y)
xx= barplot(y.frequency, col=c("blue", "yellow") ,main = "Bar plot for frequency of y", xlab = "y",ylim = c(0,40000))
text(x = xx, y = y.frequency, labels = y.frequency, pos = 3, col = "black")

education.frequency = table(bank_data$education)
xx= barplot(education.frequency, col=rainbow(4) ,main = "Bar plot for frequency of education", xlab = "education",ylim = c(0,25000))
text(x = xx, y = education.frequency, labels = education.frequency, pos = 3, col = "black")

job.frequency = table(bank_data$job)
xx= barplot(job.frequency, col=rainbow(4) ,main = "Bar plot for frequency of job",ylim = c(0,11000),las=2)
text(x = xx, y = job.frequency, labels = job.frequency, pos = 3, col = "black")

marital.frequency = table(bank_data$marital)
xx= barplot(marital.frequency, col=rainbow(4) ,main = "Bar plot for frequency of marital",xlab="marital", ylim = c(0,30000))
text(x = xx, y = marital.frequency, labels = marital.frequency, pos = 3, col = "black")

default.frequency = table(bank_data$default)
xx= barplot(default.frequency, col=rainbow(4) ,main = "Bar plot for frequency of default",xlab="default", ylim = c(0,50000))
text(x = xx, y = default.frequency, labels = default.frequency, pos = 3, col = "black")

housing.frequency = table(bank_data$housing)
xx= barplot(housing.frequency, col=rainbow(4) ,main = "Bar plot for frequency of housing",xlab="housing",ylim = c(0,27000))
text(x = xx, y = housing.frequency, labels = housing.frequency, pos = 3, col = "black")

loan.frequency = table(bank_data$loan)
xx= barplot(loan.frequency, col=rainbow(4) ,main = "Bar plot for frequency of loan",xlab="loan",ylim = c(0,40000))
text(x = xx, y = loan.frequency, labels = loan.frequency, pos = 3, col = "black")


contact.frequency = table(bank_data$contact)
xx= barplot(contact.frequency, col=rainbow(4) ,main = "Bar plot for frequency of contact",xlab="contact",ylim = c(0,33000))
text(x = xx, y = contact.frequency, labels = contact.frequency, pos = 3, col = "black")

month.frequency = table(bank_data$month)
xx= barplot(month.frequency, col=rainbow(4) ,main = "Bar plot for frequency of month",xlab="month",ylim = c(0,15000))
text(x = xx, y = month.frequency, labels = month.frequency, pos = 3, col = "black")

poutcome.frequency = table(bank_data$poutcome)
xx= barplot(poutcome.frequency, col=rainbow(4) ,main = "Bar plot for frequency of poutcome",xlab="poutcome",,ylim = c(0,40000))
text(x = xx, y = poutcome.frequency, labels = poutcome.frequency, pos = 3, col = "black")

# plot boxplot for numerical variables 
boxplot(bank_data$age, ylab="age", main = "boxplot for age",col = "blue")
text(y = boxplot.stats(bank_data$age)$stats, labels = boxplot.stats(bank_data$age)$stats, x = 1.25)


boxplot(bank_data$duration, ylab="duration", main = "boxplot for duration",col = "blue") ＃many outlier


boxplot(bank_data$day, ylab="day", main = "boxplot for day",col = "blue")
text(y = boxplot.stats(bank_data$day)$stats, labels = boxplot.stats(bank_data$day)$stats, x = 1.25)

boxplot(bank_data$campaign, ylab="campaign", main = "boxplot for campaign",col = "blue") #many outliers

boxplot(bank_data$pdays, ylab="pdays", main = "boxplot for pdays",col = "blue") # many outliers 

boxplot(bank_data$previous, ylab="previous", main = "boxplot for previous",col = "blue") # one extreme outlier


# plot histogram for numerical variables 
hist(bank_data$age, right=FALSE,main="Histogram for age", xlab="age",labels = TRUE,xlim=c(18,100),col=rainbow(10),breaks=10)

hist(bank_data$day, right=FALSE,main="Histogram for day", xlab="day",labels = TRUE,ylim = c(0,10000),col=rainbow(10),breaks=10)      

hist(bank_data$duration, right=FALSE,main="Histogram for duration", xlab="duration",labels = TRUE,ylim = c(0,41000),col=rainbow(10),breaks=10)
      
hist(bank_data$campaign, right=FALSE,main="Histogram for campaign", xlab="campaign",labels = TRUE,ylim = c(0,40000),col=rainbow(10),breaks=10)
         
hist(bank_data$pdays, right=FALSE,main="Histogram for pdays", xlab="pdays",labels = TRUE,ylim = c(0,40000),col=rainbow(10),breaks=10)

hist(bank_data$previous, right=FALSE,main="Histogram for duration", xlab="duration",labels = TRUE,ylim = c(0,50000),col=rainbow(10),breaks=10)  # one extreme outlier 275


# Check attributes correlation
numeric_bank_data=data.frame(sapply(bank_data,as.numeric))
correlationMatrix <- cor(bank_data)
# summarize the correlation matrix
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)

