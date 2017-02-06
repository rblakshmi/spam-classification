
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)
sms_raw <- read.csv("E:/rb/sms_spam.csv", stringsAsFactors=FALSE)
sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

#data preparation
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)

#lower case
sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
as.character(sms_corpus_clean[[1]])

#remove number
sms_corpus_clean <- tm_map(sms_corpus_clean , removeNumbers)

#remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean , removeWords, stopwords())

#remove punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#stemming
sms_corpus_clean <- tm_map(sms_corpus_clean , stemDocument)

#remove white space
sms_corpus_clean <- tm_map(sms_corpus_clean , stripWhitespace)


#tokenization
sms_data <- DocumentTermMatrix(sms_corpus_clean)
sms_data

#training and testing data
sms_data_train <- sms_data[1:4169 ,]
sms_data_test <- sms_data[4170:5559 ,]

sms_train_labels <- sms_raw[1:4169 ,]$type
sms_test_labels <- sms_raw[4170:5559 ,]$type

wordcloud(sms_corpus_clean , min.freq = 50, random.order = FALSE)
wordcloud(subset(sms_raw, type == "spam")$text, max.words = 40,scale = c(3,0.5))

#finding frequent words
sms_freq_words <- findFreqTerms(sms_data_train, 5)
sms_data_train_freq <- sms_data_train[, sms_freq_words]
sms_data_test_freq <- sms_data_test[, sms_freq_words]
sms_train <- apply(sms_data_train_freq , MARGIN = 2 , conver_count <- function(x) {
  x <- ifelse(x>0 , "Yes" , "No")
})
sms_test <- apply(sms_data_test_freq , MARGIN = 2 , conver_count)

#training data
sms_classifier <- naiveBayes(sms_train, sms_train_labels,laplace = 1)

# predict
sms_test_pred <- predict(sms_classifier, sms_test)
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted','actual'))
