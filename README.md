# road_sign_recognition
A machine learning project using CNN for recognising traffic signs

- To train the dataset, use this command:
py train.py --dataset dataset --model output/trafficsignnet.model --plot output/plot.png

--dataset: The dataset folder.
--model: The folder where the model is stored after training process finished.
--plot: The name of the file containing the graph of val_loss and train_loss through each epoch.

- To make prediction in the test set:
py predict.py --model output/trafficsignnet.model  --images dataset/Test --examples examples

--model: The model which is previously trained
--images: The folder containing images for prediction.
--examples: The folder where prediction results are stored.
