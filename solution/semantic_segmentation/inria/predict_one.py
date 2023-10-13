import torch


SAMPLE_PATH = "/Users/cristianion/Desktop/visual_recognition_train/inria/sample_color.jpg"
MODEL_PATH = "/Users/cristianion/Desktop/visual_recognition_train/models/inria/inria_model_1_0_0.pt"


def load_eval_model():
    model = torch.load(MODEL_PATH)
    model.eval()
    return model


def predict_one(img):
    model = load_eval_model()
    print(model)


if __name__ == "__main__":
    predict_one(SAMPLE_PATH)