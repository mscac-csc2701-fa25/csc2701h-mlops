import argparse
import mlflow
import re
import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

import config  


SETTINGS['mlflow'] = False

data_yaml=config.DATA_YAML
remote_server_uri = config.MLFLOW_SERVER

model_path="yolo11n.pt"
imgsz = 640
class_weights = [2,1]


def on_train_epoch_end(trainer):
    epoch = getattr(trainer, "epoch", 0)

    metrics = getattr(trainer, "metrics", {})
    print(metrics)
    for key, val in metrics.items():
        if isinstance(val, (float, int)):
            mlflow.log_metric(re.sub(r'\W+', '_', key), val, step=epoch)

    fitness = getattr(trainer, "fitness", None)
    if isinstance(fitness, (float, int)):
        mlflow.log_metric("fitness", fitness, step=epoch)

    tloss = getattr(trainer, "tloss", None)

    if tloss is not None:
        tloss_values = tloss.tolist() if hasattr(tloss, 'tolist') else list(tloss)
        loss_names = ["train_box_loss", "train_cls_loss", "train_dfl_loss"]
        for name, val in zip(loss_names, tloss_values):
            if isinstance(val, (float, int)):
                mlflow.log_metric(name, val, step=epoch)

def train_yolo(
        epochs, 
        run_name,
        model_path=model_path, 
        data_yaml=data_yaml, 
        imgsz=imgsz, 
        use_class_weights=None, 
        use_augmentation=False, 
    ):

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
        
    
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("fire_vs_smoke_experiments")

    if mlflow.active_run():
        mlflow.end_run()

    try:    
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model", model_path)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("imgsz", imgsz)
            mlflow.log_param("class_weights", class_weights if use_class_weights else None)
            mlflow.log_param("augmentation", use_augmentation)

            if use_augmentation:
                # Apply custom augmentation to the dataset beforehand
                pass
            # Train model
            model = YOLO(model_path) # ultralytics.engine.model.Model
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz) # 

            # Log metrics
            if hasattr(results, "results_dict"):
                for key, val in results.results_dict.items():
                    if isinstance(val, (float, int)):
                        mlflow.log_metric(re.sub(r'\W+', '_', key), val, step=epochs)
            final_tloss = getattr(model.trainer, "tloss", None)
            if final_tloss is not None:
                tloss_values = final_tloss.tolist() if hasattr(final_tloss, 'tolist') else list(final_tloss)
                loss_names = ["train_box_loss", "train_cls_loss", "train_dfl_loss"]
                for name, val in zip(loss_names, tloss_values):
                    if isinstance(val, (float, int)):
                        mlflow.log_metric(name, val, step=epochs)
                        

            print("---------------")
            print(results.results_dict)
            print(model.trainer.metrics)
            metrics = model.val()
            print(metrics.box)
            print("---------------")

            # Log artifacts
            train_dir = results.save_dir
            best_weights = os.path.join(train_dir, "weights", "best.pt")
            if os.path.exists(best_weights):
                mlflow.log_artifact(best_weights)

            mlflow.log_artifacts(train_dir) # do we need to store all artifacts yolo produces?

    except Exception as e:
        print(f"Training failed: {e}")
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise

def main():
    train_yolo(epochs=10, run_name="Default")


if __name__ == "__main__":
    # Option A: CLI mode
    if len(os.sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="yolo11n.pt")
        parser.add_argument("--data", type=str, default="data.yaml")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--experiment", type=str, default="YOLOv8-Fire-Detection")
        parser.add_argument("--run-name", type=str, default="scheduled-run")
        args = parser.parse_args()

        train_yolo(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            run_name=args.run_name
        )

    # Option B: Hardcoded or test runs
    else:
        main()

