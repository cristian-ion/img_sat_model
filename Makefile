eurosat:
	python -m solution.scene_classification.train solution/scene_classification/model_eurosat.yaml

resisc:
	python -m solution.scene_classification.train solution/scene_classification/model_resisc45.yaml

building:
	python -m solution.building_detection.train

dstl:
	python -m solution.dstl_detection.dstl_train

dstl_predict:
	python -m solution.dstl_detection.dstl_predict