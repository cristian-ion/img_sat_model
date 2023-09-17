eurosat:
	python -m solution.satellite_imagery_classification.train solution/satellite_imagery_classification/model_eurosat.yaml

resisc:
	python -m solution.satellite_imagery_classification.train solution/satellite_imagery_classification/model_resisc45.yaml

building:
	python -m solution.building_detection.train

dstl:
	python -m solution.dstl_detection.dstl_train

dstl_predict:
	python -m solution.dstl_detection.dstl_predict

dstl_eval:
	python -m solution.dstl_detection.dstl_eval

format:
	isort solution
	black solution
	flake8 --ignore=E1,E23,E203,W503,E501 solution