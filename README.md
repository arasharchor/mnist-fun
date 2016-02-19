# Noise, Model Performance, and MNIST
A weekend dive into the effects that noisy data have on model performance with MNIST. 

## The Effect of Noisy Image Data
As noise characterized by a Gaussian distribution is added to examples of different digits from the MNIST dataset, the digits become harder to distinguish (as seen below). 
![Image](/plots/Noisy_X_all_digits_0-192_with_x_and_y_label.png)

The class-wise accuracies for models trained on images with different levels of Gaussian noise is presented below. 8 is the least robust to the addition of noise, perhaps because the digit can easily "transform" into other digits. For the model trained on noisy image data with sigma=192, 8s preferentially turn into 0s (at a rate of 1.5%) and 9s (1.7%). 0s and 1s are the most robust to the addition of noise. All accuracies were calculated with models trained for 4 epochs. A rolling mean has been applied to emphasize the trends and make the plot readable.
![Image](/plots/classwise_accuracy_vs_noisy_X_jet_4.png)


## The Effect of Mislabeled Data and Accuracy Surfaces
The success of models trained on noisy labels is dependent on many hyperparameters. Here, I explore the connection between percentage of mislabeled data, batch size, dropout, and accuracy. Distinct trends emerge for models that utilize dropout and those that do not when the percentage of misclassified labels is between 10% and 60%. All models were trained for 10 epochs. 

Below is the accuracy surface for models with dropout. For a given level of label noise, increasing the batch size increases the accuracy. 
![Image](/plots/acc_grid_v2_drop_view1_highres.png)

The accuracy surface for models without dropout shows a distinctly different shape. For a given level of label noise, both small and large batch sizes outperform medium sized batches. It's likely that all of these models are overfitting to some extent, as test accuracies were higher on models trained for 4 epochs rather than 10 epochs (holding everything else constant). The small and large batches have higher bias than medium sized batches, but for slightly different reasons. With small batches, we end up performing significantly more weight updates (20x more with a batch size of 8 than with 128), and although these are inherently noisier, the larger number of updates raises the bias. For large batches (512, 1024), the bias is high in each weight update due to the large number of samples per batch. Medium sized batches (128, 256) get caught in the middleground of having high variance within each sample but not enough weight updates to raise the bias. As such, they overfit the training data and underperform on the test set. 
![Image](/plots/acc_grid_v2_nodrop_view1_highres.png)

Viewing them together shows that models with dropout have convincingly higher test set accuracies than models without dropout at all batch sizes except for 8 and 16. With high levels of label noise, a model without dropout will significantly overfit the training data, and its performance on the test set will suffer. It is unclear why models without dropout outperform models with dropout at a batch size of 8 and in some cases at a batch size of 16.

Two plots are shown, split at a batch size of 32 for clarity in emphasizing trends.
![Image](/plots/acc_grid_v2_trim_view1_highres.png)
![Image](/plots/acc_grid_v2_trim2_view1_highres.png)

