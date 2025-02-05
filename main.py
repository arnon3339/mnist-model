import argparse
from modules import model
from modules.tunning import optuna_tune as tune
import toml
import random
import builtins
import os

try:
    os.makedirs("output/models")
    os.makedirs("assets/data")
except:
    pass


random.seed(123456)

with open("config.toml", "r") as f:
    builtins.CONFIG = toml.load(f) 

if __name__ == "__main__":
    """ Train model
    """

    parser = argparse.ArgumentParser(
        prog="Drawing digit classifier",
        description="The application use CNN to classify drawing digit.",
    )
    parser.add_argument('-op', '--optimized', action='store_true', help="use optimized values")
    parser.add_argument('-f', '--fit', action='store_true', help="fit the model")
    parser.add_argument('-t', '--tune', action='store_true', help="fine-tunning the model")
    parser.add_argument('-d', '--deploy', action='store_true', help="deploy application")
    parser.add_argument('-e', '--export', action='store_true', help="export model")
    parser.add_argument('-cf', '--convert-format', action='store_true',help='model format h5 -> onnx')
    parser.add_argument('-mf', '--model-format', default="keras", type=str, help='model format (h5 or onnx)')
    parser.add_argument('-ed', '--export-data', action='store_true', help="export image data to csv dataset")
    parser.add_argument('-nt', '--number-trials', default=10, type=int, help='Number of trials for tunning')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('-nf', '--number-filters', default=32, type=int, help='Number of CNN filters')
    parser.add_argument('-dr', '--dropout-rate', default=0.2, type=float, help='Drop rate')
    args = parser.parse_args()

    if args.model_format == "lite":
        model.train.keras2tflite()
    elif args.model_format == "onnx":
        model.train.keras2onnx()
    elif args.tune:
        tune.run(args.number_trials)
    elif args.optimized:
        if args.fit:
            model.train.train_model(
                num_filters=CONFIG['optimize']['number_filters'],
                lr=CONFIG['optimize']['learning_rate'],
                dropout_rate=CONFIG['optimize']['dropout_rate'],
                export=args.model_format
            )

    # if args.export_data:
    #     data.export.image2dataset()
    # else:
    #     pass

    
    # data = pd.read_csv("./assets/data/dataset.csv", index_col=False)
    # test_img_arr = data[data['label'] == 0].iloc[0, 1:].values.astype(np.uint8)
    # print(test_img_arr)
    # img = Image.fromarray(test_img_arr.reshape((28, 28)))
    # img.save("./xxx.png")
    # with open("config.toml", "r") as f:
    #     CONFIG = toml.load(f)
    # print(CONFIG)
    # print(args.export_data)
    # os.system("python train.py")
    # os.system("python evaluate.py")
    # os.system("python tune_model.py")
    # os.system("python export_model.py")
    # os.system("uvicorn api:app --reload")  # Deploy API
