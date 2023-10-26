eurosat_train:
	python -m train.classification.train_val train/classification/config_eurosat.yaml

resisc_train:
	python -m train.classification.train_val train/classification/config_resisc45.yaml

ucm21_train:
	python -m train.classification.train_val train/classification/config_ucm21.yaml

mub_train:
	python -m train.main mu_buildings

dstl_train:
	python -m train.main dstl

inria_train:
	python -m train.main inria

inference_inria:
	python -m inference.inference_inria

format:
	isort train
	black train
	flake8 --ignore=E1,E23,E203,W503,E501 train

test:
	python -m pytest -v
