eurosat:
	python -m solution.scene_classification.train solution/scene_classification/model_eurosat.yaml

resisc:
	python -m solution.scene_classification.train solution/scene_classification/model_resisc45.yaml

dstl_train:
	python -m solution.dstl_segmentation.train solution/dstl_detection/model_resisc45.yaml