#Analisis con keras

library(keras)


mnist <- dataset_mnist()

#Entrenamiento

x_train <- mnist$train$x
y_train <- mnist$train$y

#Prueba

x_test <- mnist$test$x
y_test <- mnist$test$y


# Reescalar y reordenar

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255 
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

#Definiendo el Modelo

model <- keras_model_sequential()

model %>% layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% layer_dropout(rate = 0.4) %>% layer_dense(units = 128, activation = 'relu') %>% layer_dense(units = 10, activation = 'softmax')

#Compilar (Definir y optimizar)


model %>% compile(loss = 'categorical_crossentropy', optimizer = optimizer_rmsprop(), metrics = c('accuracy'))


#Entrenamiento

model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2)


#Evaluacion

model %>% evaluate(x_test, y_test)

model %>% predict(x_test) %>% k_argmax()

model %>% predict_on_batch(x_test) 

