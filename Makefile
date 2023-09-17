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

dstl_predict:
	python -m solution.dstl_detection.dstl_predict

dstl_eval:
	python -m solution.dstl_detection.dstl_eval

format:
	isort solution
	black solution
	flake8 --ignore=E1,E23,E203,W503,E501 solution
