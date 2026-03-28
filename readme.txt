welcome to my Suspicious Activity Detector


requied data:
https://www.dropbox.com/scl/fo/ltie6ewqb6wipoe4lk4dl/AP7Txsq-st3V45fMMCVv94g?rlkey=8381cgy7qo5e6mcjju9rpa4we&st=67bktqhv&dl=0



To set up this project, clone the repository and install the necessary dependencies using pip install -r requirements.txt.
You must manually create a data/ folder and place your labelled_train.csv, labelled_test.csv, 
and labelled_validation.csv files inside since they are excluded from the repository for privacy. Finally, 
run python train.py to execute the full pipeline, 
which uses data_loader.py to scale your features and model.py to initialize the neural network for training and evaluation.