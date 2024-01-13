Yahoo-Movies-Recommender-System

Repository Structure
--------------------

    |- Run_MLP_Rec.py           #Script for running the MLP model
    |- Vanilla_Matrix_Factorization.py # Script for running Matrix Factorization Model
    |- image_transform# scripts for image augmentations
    |- item_map.pkl   # item_map.pkl file (unique item_map values)
    |- metrics.log # Script for logging metrics
    |- pre_process.py  # Pre-processing script
    |- run_on_flask.py  # Script for Flask application
    |- trained_model.pth          # Trained Model File
    |- user_map.pkl    # user_map.pkl
    |- main.py   # scripts for pretraining on flickr30k(main process)
    |- metrics.py # scripts for loss functions, optimizer functions
    |- models.py       # backbone models and finetune models
    |- LICENSE          # license
    |- README.md        # the top level description of content and commands to reproduce results, data download instructions
    |- train_fns.py  # contains scripts for training, validation functions
    |- utils.py #   # scripts for helper functions and metrics calculation code



## GOAL

The goal is to build a recommender system and provide recommendations of Yahoo movies to users based on the interaction history of users. So initially I built a matrix factorization model as a baseline and then built a map model with embeddings and improved the model's predictions compared to the baseline model. The model outputs a rating given the user-movie interaction history

This project uses it to build a better recommendation model compared to the base model which reduces inconsistency in movie recommendation


## Results and Figures showing improvement in predictions by MLP model
Vanilla Matrix Factorization value loss

LOWEST MSE- 1.53

![Vanilla_MF](https://user-images.githubusercontent.com/93844635/210679980-01ab556a-f74e-4607-bea9-aa16f386a1e9.PNG)

MLP

LOWEST MSE- 0.94

![MLP_val_train_loss](https://user-images.githubusercontent.com/93844635/210680080-768381ce-cd65-4af3-8779-41f1f03ac03e.PNG)

To run the 'RUN_MLP_Rec.py' Function use the following command in your terminal





python3 Run_Rec.py --train train_path --test test_path --features 15 --hidden 64 --learning-rate 0.001 --batch-size 256 --epochs 100 --patience 10

Results are stored in your metrics.log file in the directory
