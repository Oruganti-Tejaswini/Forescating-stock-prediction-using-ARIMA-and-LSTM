# Arima Model
install.packages('quantmod')
install.packages('tseries')
install.packages('timeSeries')
install.packages('forecast')
installed.packages('xts')
library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
getSymbols('SPY',from='2014-01-01',to='2019-09-09')
SPY_close_Prices=SPY[,4]
# plotting 5 years data for close price
plot(SPY_close_Prices)
class(SPY_close_Prices)
par(mfrow=c(1,2))
Acf(SPY_close_Prices,main='ACF for Differenced Series')

Pacf(SPY_close_Prices, main='PACF for Differenced Series')
# p-value=0.2287
print(adf.test(SPY_close_Prices))
auto.arima(SPY_close_Prices,seasonal=FALSE)
fitA=auto.arima(SPY_close_Prices,seasonal=FALSE)


tsdisplay(residuals((fitA),lag.max=10,main='(3,1,4) Model Resediuals'))
auto.arima(SPY_close_Prices,seasonal=FALSE)
fitB=arima(SPY_close_Prices,order=c(1,2,4))
tsdisplay(residuals(fitB),lag.max=40,main='(1,2,4) model;')
fitC=arima(SPY_close_Prices,order=c(5,1,4))
tsdisplay(residuals(fitB),lag.max=40,main='(5,1,4) model;')
fitD=arima(SPY_close_Prices,order=c(1,1,1))
tsdisplay(residuals(fitB),lag.max=40,main='(1,1,1) model;')

# plots of arima model

par(mfrow=c(2,2))
term=400
fcast1<-forecast(fitA,h=term)
plot(fcast1)
fcast2<-forecast(fitB,h=term)
plot(fcast2)
fcast3<-forecast(fitC,h=term)
plot(fcast3)
fcast4<-forecast(fitD,h=term)
plot(fcast4)
accuracy(fcast2)

# --------------------------LSTM MODEL-----------------------------------------------------------------------------------------------

install.packages('quantmod')
install.packages('tseries')
install.packages('timeSeries')
install.packages('forecast')
installed.packages('xts')
install.packages("devtools")
installed.packages('tidyquant')
installed.packages('keras')
installed.packages('rsample')
devtools::install_github("rstudio/keras")
library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
library(tensorflow)
library(keras)
library(caTools)
library(caret)
library(Metrics)
getSymbols('SPY',from='2014-01-01',to='2019-09-09')
SPY_close_Prices=SPY[,4]
#creating the lags
lag_1<-Lag(SPY_close_Prices,k=1)
lag_2<-Lag(SPY_close_Prices,k=2)
lag_3<-Lag(SPY_close_Prices,k=3)
lag_4<-lag(SPY_close_Prices,k=4)
lag_5<-lag(SPY_close_Prices,k=5)
SPY_close_Prices1<-data.frame(actual=SPY_close_Prices,Lag1=lag_1,Lag2=lag_2,Lag3=lag_3,Lag4=lag_4,Lag5=lag_5)
SPY_close_Prices1<-SPY_close_Prices1[6:nrow(SPY_close_Prices1),]
SPY_close_Prices1
# normalizing the data
data_range<-function(x) {(x-min(x))/(max(x)-min(x))}
SPY_close_Prices12<-as.matrix(sapply(SPY_close_Prices1,data_range))
unscale_data<-function(x,max_x,min_x){x*(max_x-min_x)+min_x}
set_random_seed (8)
#partition of the data into test and train
train_ind = createDataPartition(SPY_close_Prices1$SPY.Close, p = 0.6, list = FALSE)
SPY_TRAIN = SPY_close_Prices12[train_ind,]
SPY_TEST = SPY_close_Prices12[-train_ind,]
actual_target_y = SPY_close_Prices1[-train_ind,1]
xkeras_train<-as.matrix(SPY_TRAIN[,2:4])
ykeras_train<-as.matrix(SPY_TRAIN[,1])
xkeras_test<-as.matrix(SPY_TEST[,2:4])
ykeras_test<-as.matrix(SPY_TEST[,1])
dim(xkeras_train)<-c(nrow(xkeras_train),ncol(xkeras_train),1)
dim(xkeras_test)<-c(nrow(xkeras_test),ncol(xkeras_test),1)
model<-keras_model_sequential()
#LSTM MODEL
model%>%
  layer_lstm(5,input_shape = c(ncol(xkeras_train),1),activation="relu")%>%
  layer_dense(units=128,activation="relu")%>%
  layer_dense(units=64,activation="relu")%>%
  layer_dense(units=1)

model%>%compile(
  loss="mae",
  optimizer="RMSprop",
  metrics=c("mae") 
)
model%>%fit(xkeras_train,ykeras_train,epochs=50,batch_size=256,shuffle=F)
y_pred=model%>%predict(xkeras_test)
lstm_actual<-unscale_data(y_pred,max(SPY_close_Prices1[,1]),min(SPY_close_Prices[,1]))
y_test = unscale_data(ykeras_test,max(SPY_close_Prices1[,1]),min(SPY_close_Prices1[,1]))
#plots of lstm model
plot(SPY_close_Prices1[-train_ind,1],type="l",ylim=c(100,300),xlim=c(100,550),xlab="time",ylab="stock price")
lines(lstm_actual,type="l",col="red")
mae(actual_target_y,lstm_actual)










