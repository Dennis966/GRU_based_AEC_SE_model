cd main


CUDA_VISIBLE_DEVICES=0 python main_SE.py --retrain \
		              --model First_SE_model_4 --loss MSE --opt Adam \
			      --epochs 3 --train_batch_size 32 \
			      --learning_rate 5e-5 \
			      --retest --test_batch_size 1

CUDA_VISIBLE_DEVICES='' python main_SE.py  \
			      --model First_SE_model_4 --loss MSE --opt Adam \
			      --epochs 3 --train_batch_size 32 \
			      --learning_rate 5e-5 \
			      --rescore
