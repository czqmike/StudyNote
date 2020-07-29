## 文本分类问题评价指标
混淆矩阵:
|                    | Classified Positive | Classified Negative |
|:------------------:|:-------------------:|:-------------------:|
|Actual Positive     |          TP         |          FN         |
|Actual Negative     |          FP         |          TN         |

where： 
 TP True Positive
 False Positive
 FN False Negative
 True Negative 

上述表格被称为Confusion Matrix

- 准确率 (Accuracy)
  `正确的样本数 / 全部样本数`
  可以用于多分类.
- 精确率 (Precision)
  表示预测的精度, 即所有预测中预测正确的占比
  `正确的正类数 / 所有被预测成正类的样本数` i.e. $p = \frac{TP} {TP + FP}$
  只能用于**二分类**
- 召回率 (Recall, Sensitivity)
  表示预测的准度, 即预测出来的一定是正确的概率.
  `正确的正类数 / 所有真正的正类样本数` i.e. $r = \frac{TP} {TP + FN}$
  只能用于**二分类**
  > e.g.
  对于地震的预报, 我们希望即使牺牲Precision, Recall尽可能高, 即情愿发出1000次警报, 把10次地震都预报出来了, 也不希望100次漏了2次.
  对于犯罪嫌疑人, 遵循无罪推定, 不错杀一个好人的原则. 即使Recall率低, 也是值得的.
- F1值 (F1-score)
  P/R的调和均值
  $\frac{2}{F_1} = \frac1{P} + \frac1{R}$
- Specificity (Ture Negative Rate, TNR)
  特异度, Recall的相反表达
- False Positive Ratio (FPR)
- False Negative Ratio (FNR)
- AUC (the Area Under an ROC Curve) 
  ROC曲线就是以FPR为x轴，TPR为y轴画图得到。
  AUC就是ROC曲线的面积.