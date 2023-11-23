# Visual recognition train
# Sat Img Model

Segmentation:
- [Massachusetts Roads Dataset, Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/)
- [Inria Dataset](https://project.inria.fr/aerialimagelabeling/)
- [DSTL Dataset](https://www.kaggle.com/competitions/dstl-satellite-imagery-feature-detection/overview)
- [Papers with code: Semantic Segmentation on INRIA Aerial Image Labeling](https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image)
- [Building Extraction from Remote Sensing Images via an Uncertainty-Aware Network](https://arxiv.org/pdf/2307.12309v1.pdf)

Classification:
- EuroSat:
- Resisc45:
- UC Merced Land use:

Detection:
- Mask R-CNN
- Window method
- DOTAv2 dataset
- Dior dataset

# On windows:
.venv\\Scripts\\activate

# On Linux / MAC
source .venv/bin/activate


# Run inference:
`python`\
`>>> from inference.inference_inria import InferenceInria`\
`>>> inference = InferenceInria(debug=False, save_out=True)`\
`>>> inference.infer_file("/Users/cristianion/Desktop/img_sat_model/inria/sample_color.jpg")`


Start notify script:
`while true; do python notify.py; sleep 3600; done`

Models need to be copied to the models folder under this structure: `models/inria/inria_model_1_0_7.pt` in order
to be availalbe for inference.