# Emotion-Classification-Comparison
Classification comparison between machine learning models and techniques on emotion data-set. You can download the dataset [here](https://github.com/huseinzol05/NLP-Dataset)

### Vectorization techniques I used:
1. [Bag Of Word / Unigram](https://en.wikipedia.org/wiki/Bag-of-words_model)
2. [Tfidf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
3. Timestamp based on dictionary position
4. [SVD / LSA](http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm)
5. [Word Vector](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)
6. Hashing Vectorization

### Models that I used:
1. Extreme Gradient Boosting
2. Light Gradient Boosting
3. Feed-forward Neural Network
4. Recurrent Neural Network + LSTM
5. Convolutional Neural Network
6. Convolutional + Recurrent
7. Ensemble methods
8. Naive Bayes
9. SVM
10. Bidirectional Recurrent Neural Network + LSTM
11. Recurrent Neural Network + LSTM + Huber
12. Recurrent Neural Network + LSTM + Hinge (SVM)

#### All the notebooks applied pre-processing text cleaning using Regex. re.sub('[^A-Za-z0-9 ]+', '', string).
#### All the models applied early-stopping to prevent overfit.
#### Assuming BOW and TFIDF generated all are the same.
#### All the models trained 80% of the dataset, validated 20% of the dataset.
#### Some comparisons are not consistent, example in Neural Network based, I do not calculate recall, and f1. Need to update later.

## BOW / Unigram
1. [Naive Bayes](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/bayes-classifier.ipynb)
```text
accuracy validation set:  0.859072479067
             precision    recall  f1-score   support

      anger       0.90      0.84      0.87     11464
       fear       0.84      0.81      0.82      9455
        joy       0.85      0.93      0.89     28246
       love       0.82      0.61      0.70      6920
    sadness       0.87      0.94      0.91     24263
   surprise       0.84      0.34      0.49      3014

avg / total       0.86      0.86      0.85     83362
```
2. [SVM Kernel based](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/stochastic-classifiers.ipynb)
```text
accuracy validation set:  0.898586886111
             precision    recall  f1-score   support

      anger       0.91      0.88      0.90     11422
       fear       0.84      0.87      0.86      9495
        joy       0.90      0.94      0.92     28138
       love       0.84      0.74      0.79      6970
    sadness       0.93      0.94      0.94     24380
   surprise       0.85      0.65      0.73      2957

avg / total       0.90      0.90      0.90     83362
```
3. [XGB](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/xgb-bow.ipynb)
```text
accuracy validation set:  0.895132074566
             precision    recall  f1-score   support

      anger       0.88      0.92      0.90     11421
       fear       0.83      0.84      0.84      9505
        joy       0.93      0.91      0.92     28132
       love       0.76      0.79      0.78      6801
    sadness       0.95      0.94      0.94     24481
   surprise       0.70      0.72      0.71      3022

avg / total       0.90      0.90      0.90     83362
```

## TFIDF
1. [Naive bayes](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/bayes-classifier.ipynb)
```text
accuracy validation set:  0.734855209808
             precision    recall  f1-score   support

      anger       0.93      0.54      0.69     11336
       fear       0.91      0.37      0.53      9603
        joy       0.68      0.98      0.80     28062
       love       0.96      0.16      0.27      7085
    sadness       0.74      0.94      0.83     24278
   surprise       0.94      0.04      0.08      2998

avg / total       0.79      0.73      0.69     83362
```
2. [SVM Kernel based](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/stochastic-classifiers.ipynb)
```text
accuracy validation set:  0.850915285142
             precision    recall  f1-score   support

      anger       0.93      0.75      0.83     11542
       fear       0.88      0.73      0.79      9610
        joy       0.79      0.97      0.87     28110
       love       0.92      0.55      0.69      6883
    sadness       0.88      0.94      0.91     24230
   surprise       0.91      0.46      0.61      2987

avg / total       0.86      0.85      0.84     83362
```
3. [XGB](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/xgb-tfidf.ipynb)
```text
accuracy validation set:  0.885415417097
             precision    recall  f1-score   support

      anger       0.88      0.90      0.89     11414
       fear       0.81      0.83      0.82      9584
        joy       0.92      0.91      0.91     28269
       love       0.74      0.77      0.75      6878
    sadness       0.95      0.93      0.94     24196
   surprise       0.66      0.67      0.67      3021

avg / total       0.89      0.89      0.89     83362
```
4. [LGB](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/lgb-tfidf.ipynb)
```text
accuracy validation set:  0.902245627504
             precision    recall  f1-score   support

      anger       0.91      0.91      0.91     11550
       fear       0.84      0.89      0.86      9455
        joy       0.92      0.92      0.92     28299
       love       0.77      0.88      0.82      6910
    sadness       0.96      0.92      0.94     24111
   surprise       0.82      0.70      0.75      3037

avg / total       0.90      0.90      0.90     83362
```

## Timestamp based on Dictionary position
1. [LGB average word length](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/lgb-timestamp.ipynb)
```text
precision    recall  f1-score   support

      anger       0.53      0.22      0.31     11587
       fear       0.50      0.18      0.27      9504
        joy       0.46      0.73      0.57     28074
       love       0.30      0.08      0.13      6949
    sadness       0.49      0.57      0.53     24293
   surprise       0.26      0.09      0.13      2955

avg / total       0.46      0.47      0.43     83362
```
2. [RNN average word length](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/rnn-timestamp.ipynb)
```text
epoch: 74 , training loss:  1.56926864815 , train acc:  0.344564461275
epoch: 75 , training loss:  1.56929268564 , train acc:  0.345266404707
epoch: 76 , training loss:  1.56907495159 , train acc:  0.344702450399
epoch: 77 , training loss:  1.56922572671 , train acc:  0.344630455659
epoch: 78 , training loss:  1.56905308527 , train acc:  0.345173412831
epoch: 79 , training loss:  1.56924159172 , train acc:  0.34504442315
```
3. [Self optimized Feed-forward Neural Network average word length](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/self-optimized-feedforward-timestamp.ipynb)
```text
epoch: 1 , pass acc: 0.305403 , current acc: 0.335717
epoch: 2 , pass acc: 0.335717 , current acc: 0.343142
epoch: 3 , pass acc: 0.343142 , current acc: 0.345385
epoch: 4 , pass acc: 0.345385 , current acc: 0.345745
epoch: 5 , pass acc: 0.345745 , current acc: 0.346273
epoch: 8 , pass acc: 0.346273 , current acc: 0.347029
break epoch: 107
```
4. [XGB average word length](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/xgb-timestamp-avg.ipynb)
```text
precision    recall  f1-score   support

      anger       0.48      0.22      0.31     11390
       fear       0.46      0.20      0.28      9759
        joy       0.48      0.72      0.58     27981
       love       0.26      0.08      0.12      6838
    sadness       0.49      0.58      0.53     24395
   surprise       0.21      0.07      0.11      2999

avg / total       0.45      0.47      0.44     83362
```
5. [XGB 50 word length](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/xgb-timestamp50.ipynb)
```text
precision    recall  f1-score   support

      anger       0.48      0.21      0.30     11320
       fear       0.45      0.18      0.25      9658
        joy       0.47      0.72      0.57     28342
       love       0.27      0.08      0.12      6901
    sadness       0.48      0.57      0.52     24103
   surprise       0.22      0.07      0.11      3038

avg / total       0.45      0.47      0.43     83362
```

## SVD / LSA
1. [XGB 50 dimensions](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/xgb-tfidf-svd50.ipynb)
```text
precision    recall  f1-score   support

      anger       0.34      0.07      0.11     11336
       fear       0.30      0.06      0.10      9694
        joy       0.46      0.73      0.56     28068
       love       0.17      0.01      0.03      6987
    sadness       0.40      0.54      0.46     24277
   surprise       0.07      0.00      0.01      3000

avg / total       0.37      0.42      0.35     83362
```
2. [LGB 50 dimensions](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/lgb-tfidf-svd50.ipynb)
```text
precision    recall  f1-score   support

      anger       0.38      0.05      0.09     11460
       fear       0.32      0.06      0.10      9545
        joy       0.44      0.73      0.55     28052
       love       0.17      0.01      0.02      7015
    sadness       0.39      0.54      0.45     24291
   surprise       0.09      0.01      0.01      2999

avg / total       0.37      0.42      0.34     83362
```

## Word Vector
1. [Feed-forward word vector](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/feedforward-vector.ipynb)
```text
epoch: 220 , training loss: 0.536347100064 , training acc: 0.88623161306 , valid loss: 0.796489044223 , valid acc: 0.808128580994
epoch: 221 , training loss: 0.535965369 , training acc: 0.886359573154 , valid loss: 0.796616834379 , valid acc: 0.808147773232
epoch: 222 , training loss: 0.535587115036 , training acc: 0.886493929708 , valid loss: 0.796744917725 , valid acc: 0.808205356811
epoch: 223 , training loss: 0.535212274276 , training acc: 0.886564308321 , valid loss: 0.796873355467 , valid acc: 0.808291729718
break epoch: 223
```
2. [Recurrent Neural Network LSTM](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/rnn-vector.ipynb)
```text
epoch: 43 , training loss: 0.293494431497 , training acc: 0.91647871451 , valid loss: 0.301500787934 , valid acc: 0.911824742285
'unwarrentedly'
time taken: 201.13964009284973
epoch: 44 , training loss: 0.292884760499 , training acc: 0.916391732865 , valid loss: 0.316479788662 , valid acc: 0.908499411174
'unwarrentedly'
time taken: 201.1136453151703
epoch: 45 , training loss: 0.292009347957 , training acc: 0.916490714602 , valid loss: 0.301596783737 , valid acc: 0.910792332308
```
3. [Convolutional Neural Network](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/cnn-vector.ipynb)
```text
epoch: 45 , training loss: 0.406819455151 , training acc: 0.900872836373 , valid loss: 0.413255342881 , valid acc: 0.896866754467
epoch: 46 , training loss: 0.406384702676 , training acc: 0.900506910515 , valid loss: 0.412755171971 , valid acc: 0.896986806521
epoch: 47 , training loss: 0.406851520408 , training acc: 0.901097791847 , valid loss: 0.414064356509 , valid acc: 0.895942385028
```
4. [CNN + RNN](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/cnn-rnn-vector.ipynb)
```text
epoch: 11 , training loss: 0.89633845398 , training acc: 0.742852410318 , valid loss: 1.31553278515 , valid acc: 0.62818725459
'unwarrentedly'
epoch: 12 , training loss: 0.870958457021 , training acc: 0.751430695276 , valid loss: 1.35667084581 , valid acc: 0.625294097582
'unwarrentedly'
epoch: 13 , training loss: 0.867803699844 , training acc: 0.751742633378 , valid loss: 1.37888059403 , valid acc: 0.626062403385
break epoch: 13
```
5. [Bidirectional Recurrent Neural Network LSTM](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/bidirectional-rnn-vector.ipynb)
```text
epoch: 31 , training loss: 0.306716536762 , training acc: 0.915449927471 , valid loss: 0.305131908171 , valid acc: 0.911728706609
'unwarrentedly'
time taken: 633.0987968444824
epoch: 32 , training loss: 0.30510045163 , training acc: 0.915593898146 , valid loss: 0.306183017638 , valid acc: 0.911968804136
'unwarrentedly'
time taken: 633.1091139316559
epoch: 33 , training loss: 0.304126529089 , training acc: 0.915686878639 , valid loss: 0.305108016744 , valid acc: 0.911548635682
```

## Hashing Vectorization
1. [Naive Bayes](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/bayes-classifier.ipynb)
```text
accuracy validation set:  0.578524987404
             precision    recall  f1-score   support

      anger       0.93      0.07      0.12     11449
       fear       0.96      0.03      0.05      9533
        joy       0.49      1.00      0.66     28047
       love       1.00      0.00      0.01      6967
    sadness       0.76      0.79      0.78     24408
   surprise       0.00      0.00      0.00      2958

avg / total       0.71      0.58      0.47     83362
```
2. [SVM Kernel Based](https://github.com/huseinzol05/Emotion-Classification-Comparison/blob/master/stochastic-classifiers.ipynb)
```text
accuracy validation set:  0.791163839639
             precision    recall  f1-score   support

      anger       0.92      0.64      0.76     11592
       fear       0.87      0.59      0.70      9557
        joy       0.71      0.97      0.82     28068
       love       0.94      0.40      0.56      6933
    sadness       0.83      0.90      0.87     24273
   surprise       0.91      0.34      0.49      2939

avg / total       0.82      0.79      0.78     83362
```
