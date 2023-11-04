from inference.inference_inria import InferenceInria


SAMPLE_PATH = (
    "/Users/cristianion/Desktop/img_sat_model/inria/sample_color.jpg"
)


if __name__ == "__main__":
    inference = InferenceInria(debug=True, save_out=True)
    inference.infer_file(SAMPLE_PATH)
