# On separate trains, not weighted:
#Final score of cross-val: F1=0.8545034642032333, Recall = 0.8409090909090909, Precision=0.8685446009389671
#Final score of cross-val: F1=0.8663594470046083, Recall = 0.8355555555555556, Precision=0.8995215311004785
#Final score of cross-val: F1=0.8379629629629629, Recall = 0.8302752293577982, Precision=0.8457943925233645
#Final score of cross-val: F1=0.8257756563245824, Recall = 0.7972350230414746, Precision=0.8564356435643564
#Final score of cross-val: F1=0.8374164810690423, Recall = 0.8138528138528138, Precision=0.8623853211009175
# F1 = 0.844  # With no synthetic: 0.8433 # No synth and no calculated features: 0.829

# not weighted synthetic on united train:
#Final score of cross-val: F1=0.8159645232815964, Recall = 0.7965367965367965, Precision=0.8363636363636363
#Final score of cross-val: F1=0.851063829787234, Recall = 0.8181818181818182, Precision=0.8866995073891626
#Final score of cross-val: F1=0.8306264501160093, Recall = 0.7955555555555556, Precision=0.8689320388349514
#Final score of cross-val: F1=0.8443396226415094, Recall = 0.8211009174311926, Precision=0.8689320388349514
# F1 = 0.834

# Weighted synthetic on united train:
# Final score of cross-val: F1=0.8393285371702638, Recall = 0.8064516129032258, Precision=0.875
#Final score of cross-val: F1=0.8351648351648352, Recall = 0.8225108225108225, Precision=0.8482142857142857
#Final score of cross-val: F1=0.8403755868544601, Recall = 0.8136363636363636, Precision=0.8689320388349514
#Final score of cross-val: F1=0.8466819221967964, Recall = 0.8222222222222222, Precision=0.8726415094339622
#F1 = 0.84

#Weighted on separate trains:
#Final score of cross-val: F1=0.8433179723502304, Recall = 0.8394495412844036, Precision=0.8472222222222222
#Final score of cross-val: F1=0.8658823529411764, Recall = 0.847926267281106, Precision=0.8846153846153846
#Final score of cross-val: F1=0.8232662192393736, Recall = 0.7965367965367965, Precision=0.8518518518518519
#Final score of cross-val: F1=0.8436018957345972, Recall = 0.8090909090909091, Precision=0.8811881188118812
# F1 = 0.8435

# Only generate ~200 samples of 0 label
# Final score of cross-val: F1=0.8435374149659864, Recall = 0.8266666666666667, Precision=0.8611111111111112
# Final score of cross-val: F1=0.8325358851674641, Recall = 0.7981651376146789, Precision=0.87
# Final score of cross-val: F1=0.8274231678486997, Recall = 0.8064516129032258, Precision=0.8495145631067961
# Final score of cross-val: F1=0.8181818181818182, Recall = 0.7792207792207793, Precision=0.861244019138756
# F1 = 0.8367

# NEXT go the experiments with snapshots.

# With external factors (snapshot step 5m): F1 = 0.69
# test on 2024:
# F1 = 0.5888501742160279, Recall = 0.5044776119402985, Precision = 0.7071129707112971
# F1 = 0.6875, Recall = 0.6994219653179191, Precision = 0.6759776536312849
# F1 = 0.5052631578947369, Recall = 0.40955631399317405, Precision = 0.6593406593406593

# With external factors (step 3 m): F1 = 0.74
# test on 2024:
# F1 = 0.6466575716234653, Recall = 0.7074626865671642, Precision = 0.5954773869346733
# F1 = 0.6870588235294117, Recall = 0.8439306358381503, Precision = 0.5793650793650794
# F1 = 0.5167652859960552, Recall = 0.447098976109215, Precision = 0.6121495327102804

# Same but k_folds with row order preserved:
# F1 = 0.7
# F1 = 0.6147308781869688, Recall = 0.6477611940298508, Precision = 0.5849056603773585
# F1 = 0.6700507614213198, Recall = 0.7630057803468208, Precision = 0.5972850678733032
# F1 = 0.5025125628140703, Recall = 0.5119453924914675, Precision = 0.4934210526315789


#Without external factors:
# F1 = 0.58
# test on 2024:
# F1 = 0.6143344709897611, Recall = 0.5373134328358209, Precision = 0.7171314741035857
# F1 = 0.6923076923076923, Recall = 0.6763005780346821, Precision = 0.7090909090909091
# F1 = 0.494279176201373, Recall = 0.36860068259385664, Precision = 0.75

# Same but k_folds with row order preserved:
# F1 = 0.6
# F1 = 0.6127527216174183, Recall = 0.5880597014925373, Precision = 0.6396103896103896
# F1 = 0.7154929577464789, Recall = 0.7341040462427746, Precision = 0.6978021978021978
# F1 = 0.5019762845849802, Recall = 0.4334470989761092, Precision = 0.596244131455399

# Rand snap x2 with external factors, k_folds with row order preserved:: this way, finally external factors are NOT dominating by importance
# Final score of cross-val: F1=0.6860099039347485, Recall = 0.6105416850580093, Precision=0.7833616988900067
# test on 2024:
# F1 = 0.577391304347826, Recall = 0.4955223880597015, Precision = 0.6916666666666667
# F1 = 0.6884272997032641, Recall = 0.6705202312138728, Precision = 0.7073170731707317
# F1 = 0.49287169042769857, Recall = 0.4129692832764505, Precision = 0.6111111111111112

# Same without external:
# Final score of cross-val: F1=0.6922055890292269, Recall = 0.6379411185156659, Precision=0.7575995008588249
# F1 = 0.5951219512195122, Recall = 0.5462686567164179, Precision = 0.6535714285714286
# F1 = 0.6951566951566952, Recall = 0.7052023121387283, Precision = 0.6853932584269663
# F1 = 0.4812623274161736, Recall = 0.41638225255972694, Precision = 0.5700934579439252

# same with SMOTE:
#F1 = 0.77
# F1 = 0.6365054602184087, Recall = 0.608955223880597, Precision = 0.6666666666666666
# F1 = 0.6991404011461319, Recall = 0.7052023121387283, Precision = 0.6931818181818182
# F1 = 0.5259391771019678, Recall = 0.5017064846416383, Precision = 0.5526315789473685

# External + randx2 + SMOTE:
# F1 = 0.6141975308641975, Recall = 0.5940298507462687, Precision = 0.6357827476038339
# F1 = 0.6816901408450704, Recall = 0.6994219653179191, Precision = 0.6648351648351648
# F1 = 0.5106382978723404, Recall = 0.49146757679180886, Precision = 0.5313653136531366

# Add processed duplicates:
# F1 = 0.71
#F1 = 0.62, Recall = 0.58, Precision = 0.67
#F1 = 0.69, Recall = 0.71, Precision = 0.68
#F1 = 0.48, Recall = 0.40, Precision = 0.60
#Test tot: F1 = 0.59, Recall = 0.54, Precision = 0.65
