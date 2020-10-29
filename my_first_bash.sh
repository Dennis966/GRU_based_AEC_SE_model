
cd main


CUDA_VISIBLE_DEVICES=0 python main_AEC.py --retrain \
				--model First_AEC_model_4 --loss MSE --opt Adam \
				--epochs 3 --train_batch_size 32 \
				--learning_rate 4e-5 \
				--retest --test_batch_size 1


CUDA_VISIBLE_DEVICES='' python main_AEC.py  \
				--model First_AEC_model_4 --loss MSE --opt Adam \
				--epochs 3 --train_batch_size 32 \
				--learning_rate 4e-5 \
				--rescore

CUDA_VISIBLE_DEVICES=0 python main_SE_fixAEC.py --retrain \
				--model First_SE_fix_AEC_model_4 --loss MSE --opt Adam \
				--epochs 3 --train_batch_size 32 \
				--learning_rate 4e-5 \
				--retest --test_batch_size 1

CUDA_VISIBLE_DEVICES='' python main_SE_fixAEC.py \
				--model First_SE_fix_AEC_model_4 --loss MSE --opt Adam \
				--epochs 3 --train_batch_size 32 \
				--learning_rate 4e-5 \
				--rescore


