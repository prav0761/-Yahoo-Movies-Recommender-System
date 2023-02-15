Vanilla Matrix Factorization val loss

LOWEST MSE- 1.53

![Vanilla_MF](https://user-images.githubusercontent.com/93844635/210679980-01ab556a-f74e-4607-bea9-aa16f386a1e9.PNG)

MLP

LOWEST MSE- 0.94

![MLP_val_train_loss](https://user-images.githubusercontent.com/93844635/210680080-768381ce-cd65-4af3-8779-41f1f03ac03e.PNG)

To run 'RUN_MLP_Rec.py' Function use following command in your terminal





python3 Run_Rec.py --train train_path --test test_path --features 15 --hidden 64 --learning-rate 0.001 --batch-size 256 --epochs 100 --patience 10
