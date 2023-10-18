eurosat_train:
	python -m solution.classification.train_val solution/classification/config_eurosat.yaml

resisc_train:
	python -m solution.classification.train_val solution/classification/config_resisc45.yaml

ucm21_train:
	python -m solution.classification.train_val solution/classification/config_ucm21.yaml

mub_train:
	python -m solution.semantic_segmentation.train_val mub

dstl_train:
	python -m solution.semantic_segmentation.train_val dstl

inria_train:
	python -m solution.semantic_segmentation.train_val inria

inria_sample:
	python -m solution.semantic_segmentation.inria.predict_one

inria_convert_model:
	python -m solution.semantic_segmentation.convert_model

format:
	isort solution
	black solution
	flake8 --ignore=E1,E23,E203,W503,E501 solution

test:
	python -m pytest -v