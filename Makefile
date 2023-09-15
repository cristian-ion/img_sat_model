eurosat:
	python -m model.scene_classification.train solution/scene_classification/model_eurosat.yaml

resisc:
	python -m model.scene_classification.train solution/scene_classification/model_resisc45.yaml

building:
	python -m model.building_detection.train

dstl:
	python -m model.dstl_detection.dstl_train

dstl_predict:
	python -m model.dstl_detection.dstl_predict

dstl_eval:
	python -m model.dstl_detection.dstl_eval