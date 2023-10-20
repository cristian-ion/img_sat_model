import torch

MODEL_PATH = "/Users/cristianion/Desktop/visual_recognition_train/models/inria/inria_model_1_0_0.pt"
OUT_PATH = "/Users/cristianion/Desktop/visual_recognition_train/models/inria/inria_model_scripted_1_0_0.pt"


def convert_model():
    model = torch.load(MODEL_PATH)
    model_scripted = torch.jit.script(model)
    model_scripted.save(OUT_PATH)


if __name__ == "__main__":
    convert_model()
