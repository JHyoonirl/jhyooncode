import ultralytics

from roboflow import Roboflow

rf = Roboflow(api_key="HFkCToeQS6j19qszUI3y")
project = rf.workspace("jhyoon-zf6gn").project("sulivan-model")
version = project.version(1)
dataset = version.download("yolov8")


version.deploy("yolov8", "model", "best_240502_epoch50.pt")

