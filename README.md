Install dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data
(rename folder to '140k_dataset')

WSL + Ubuntu setup + VSCODE extension (cuda + cudnn install)
(Python 3.12.3, Nvidia RTX 3070 laptop)

Ubuntu->Linux->Home->[user]->project
vs code: connect to WSL folder

create .venv
source .venv/bin/activate

Make sure requirements in requirements.txt are installed

If GPU is not recognized try before running python file: https://dev.to/metal3d/how-to-resolve-the-dlopen-problem-with-nvidia-and-pytorch-or-tensorflow-inside-a-virtual-env-181e

densenet.py: DenseNet121 model can get best_densenet121.keras after training
inception.py: InceptionResNetV2 model can get best_inceptionresnet.keras after training

predict_example.py: Predict example test image (can be changed), need model to be loaded
