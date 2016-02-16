# Noise, Model Performance, and MNIST
A weekend dive into the effects that noisy data have on model performance with MNIST. 

## The Effect of Noisy Image Data
As noise characterized by a Gaussian distribution is added to examples of different digits from the MNIST dataset, the digits become harder to distinguish (as seen below). 
![Image](/plots/Noisy_X_all_digits_0-192_with_x_and_y_label.png)

The class-wise accuracies for models trained on images with different levels of Gaussian noise is presented below. 8 is the least robust to the addition of noise, perhaps because the digit can easily "transform" into other digits. For the model trained on noisy image data with sigma=192, 8s preferentially turn into 0s (at a rate of 1.5%) and 9s (1.7%). 0s and 1s are the most robust to the addition of noise. All accuracies were calculated with models trained for 4 epochs. A rolling mean has been applied to emphasize the trends and make the plot readable.
![Image](/plots/classwise_accuracy_vs_noisy_X_jet_4.png)


## The Effect of Mislabeled Data
The model accuracy on the test set (blue) for models trained with increasing levels of randomly labeled data is shown below. The dotted black line shows the accuracy achieved (0.1) by consistently guessing one value. The model performs relatively well even when ~20% of the data are mislabeled but is unable to learn anything if ~25% of the data are mislabeled. Accuracies were calculated with models trained for 4 epochs.
![Image](/plots/accuracy_vs_noisy_y_fixed_x_label.png)
