train_eurosat:
	python -m train.classification.train_val train/classification/config_eurosat.yaml

train_resisc:
	python -m train.classification.train_val train/classification/config_resisc45.yaml

train_ucm21:
	python -m train.classification.train_val train/classification/config_ucm21.yaml

train_mub:
	python -m train.main mu_buildings

train_dstl:
	python -m train.main dstl

train_inria:
	python -m train.main inria

sample_inria:
	python -m inference.inference_inria

infer_inria:
	python -m test.infer_inria_val_test

compare_inria:
	python -m test.compare_inria > models/inria/inria_model_1_0_4_scores.txt

crops_inria:
	python -m crops_inria

format:
	isort train
	black train
	flake8 --ignore=E1,E23,E203,W503,E501 train

test:
	python -m pytest -v
