import argparse
import os
import sys
import logging
import pathlib
import random
import yaml
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold

# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Local import
import dataset
import utils
from model import DeepNetwork

# Deepcs
import deepcs.display


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--logname", type=str, default=None)
    parser.add_argument("--logdir", type=pathlib.Path, default="./logs")
    return parser.parse_args()


def collate_fn(samples):
    data = [sample["input"] for sample in samples]
    labels = [sample["target"] for sample in samples]
    padded_data = pad_sequence(data, batch_first=True)
    padded_label = torch.tensor(labels)
    return {"input": padded_data, "target": padded_label}


def save_metrics_to_csv(metrics, filepath="accuracies.csv"):
    df = pd.DataFrame([metrics])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, mode="w", header=True, index=False)


def evaluate(model, loader, criterion, optimizer, device, classes):
    model.eval()
    total_loss = 0
    num_samples = 0
    correct = 0
    f1 = 0
    y_pred = []
    y_true = []
    with tqdm(
        loader, total=len(loader.dataset), desc="Validation", leave=False
    ) as pbar:
        for data in loader:
            inputs, targets = data["input"], data["target"]
            inputs, targets = inputs.to(device), targets.to(device)
            # Compute the forward propagation
            outputs = model(inputs)
            out = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            label = targets.data.cpu().numpy()
            y_pred.extend(out)
            y_true.extend(label)
            predicted_outputs = outputs.argmax(dim=1)
            correct += (predicted_outputs == targets).sum().item()
            loss = criterion(outputs.squeeze(1), targets)
            pbar.set_postfix(**{"valid_loss": loss.item()})
            pbar.update(inputs.shape[0])
            accuracy = metrics.accuracy_score(
                targets.detach().cpu().numpy().tolist(),
                outputs.argmax(dim=1).detach().cpu().numpy().tolist(),
            )
            f1 = metrics.f1_score(
                targets.detach().cpu().numpy().tolist(),
                outputs.argmax(dim=1).detach().cpu().numpy().tolist(),
                average="micro",
            )
            # Update the metrics
            # We here consider the loss is batch normalized
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]
    y_true_str = [classes[i] for i in y_true]
    y_pred_str = [classes[i] for i in y_pred]
    cf_matrix = metrics.confusion_matrix(
        y_true_str, y_pred_str, labels=classes
    )

    norm_coeff = np.sum(cf_matrix, axis=1)
    norm_coeff[norm_coeff == 0] = 1
    per_class_accuracy = np.diag(cf_matrix) / np.sum(cf_matrix, axis=1)
    per_class_accuracy_dict = {
        class_name: acc for class_name, acc in zip(classes, per_class_accuracy)
    }

    df_cm = pd.DataFrame(
        cf_matrix / norm_coeff[:, None],
        index=classes,
        columns=classes,
    )
    return total_loss / num_samples, {
        "Accuracy": correct / num_samples,
        "F1": f1,
        "CM": df_cm,
        "Per_class_accuracy": per_class_accuracy_dict,
    }


def main(args):
    global_step = 0
    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    if args.logname is None:
        logdir = utils.generate_unique_logpath(
            args.logdir, cfg["Model"]["Name"]
        )
    else:
        logdir = args.logdir / args.logname

    utils.seed_torch(seed=cfg["Training"]["SEED"])
    logdir = pathlib.Path(logdir)
    logging.basicConfig(
        filename=logdir / "result.log",
        level=logging.INFO,
        format="%(message)s",
    )
    logging.info(f"Logging into {logdir}")
    if not logdir.exists():
        logdir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindataset_cls = cfg["TrainDataset"]["cls"]
    traindataset_args = cfg["TrainDataset"]["args"]
    if traindataset_args is not None:
        # train_data = getattr(dataset, traindataset_cls)(**traindataset_args)
        train_data = eval(f"dataset.{traindataset_cls}(**traindataset_args)")
    else:
        train_data = eval(f"dataset.{traindataset_cls}()")
    # Get all the class names
    classes = train_data.classes

    # Build up the validset either from a random split from the training data
    # or from a split of the whole training set
    if "fromsplit" in cfg["ValidDataset"]:
        whole_train_data = train_data
        idx = list(range(len(whole_train_data)))
        random.shuffle(idx)
        valid_ratio = float(cfg["ValidDataset"]["fromsplit"])
        num_valid = int(valid_ratio * len(whole_train_data))
        train_idx = idx[num_valid:]
        valid_idx = idx[:num_valid]
        train_data = torch.utils.data.Subset(whole_train_data, train_idx)
        valid_data = torch.utils.data.Subset(whole_train_data, valid_idx)
    else:
        validdataset_cls = cfg["ValidDataset"]["cls"]
        validdataset_args = cfg["ValidDataset"]["args"]
        if validdataset_args is not None:
            valid_data = eval(
                f"dataset.{validdataset_cls}(**validdataset_args)"
            )
        else:
            valid_data = eval(f"dataset.{validdataset_cls}()")

    k_folds = 5
    kf = KFold(
        n_splits=k_folds, shuffle=True, random_state=cfg["Training"]["SEED"]
    )
    results = []

    for fold, (train_ids, valid_ids) in enumerate(kf.split(train_data)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg["Data"]["Batch_size"],
            sampler=train_subsampler,
            collate_fn=collate_fn,
        )
        valid_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg["Data"]["Batch_size"],
            sampler=valid_subsampler,
            collate_fn=collate_fn,
        )

        model = DeepNetwork(num_classes=len(classes), requires_grad=True)

        model.to(device)
        criterion = utils.get_criterion(cfg)
        optimizer = utils.get_optimizer(cfg, model)
        summary_text = (
            f"Logdir : {logdir}\n"
            + "## Command \n"
            + " ".join(sys.argv)
            + "\n\n"
            + f" Arguments : {args}"
            + "\n\n"
            + "## Datasets : \n"
            + f"Train : {train_loader.dataset}\n"
            + f"Validation : {valid_loader.dataset}\n"
            + "## Training params :\n\n"
            + f"Model : {model}\n\n"
            + f"Loss : {criterion}\n"
            + f"Optimizer : {optimizer}\n"
            + f"Learning rate : {cfg['Optimizer']['lr']}\n"
            + f"Batch_size : {cfg['Data']['Batch_size']}\n"
            + f"Epochs : {cfg['Training']['Epochs']}\n"
        )
        # tensorboard_writer_train = SummaryWriter(
        #     log_dir=logdir / "train", comment=" Train  "
        # )
        # tensorboard_writer_valid = SummaryWriter(
        #     log_dir=logdir / "valid", comment=" Valid  "
        # )
        # tensorboard_writer_train.add_text(
        #     "Experiment summary", deepcs.display.htmlize(summary_text)
        # )
        # tensorboard_writer_valid.add_text(
        #     "Experiment summary", deepcs.display.htmlize(summary_text)
        # )
        best_model_path = os.path.join(logdir, "best_model.bin")
        with open(logdir / "summary.txt", "w") as f:
            f.write(summary_text)

        with open(logdir / f"{args.config.split('/')[-1]}", "w") as f:
            yaml.dump(cfg, f)

        with open("model.py", "r") as f:
            fs = [line.rstrip("\n") for line in f.readlines()]

        with open(logdir / "model.py", "w") as f:
            for line in fs:
                f.write(line + "\n")

        with open("dataset.py", "r") as f:
            fs = [line.rstrip("\n") for line in f.readlines()]

        with open(logdir / "dataset.py", "w") as f:
            for line in fs:
                f.write(line + "\n")

        best_results = 0
        all_true_labels = []
        all_predicted_labels = []

        epoch_bar = tqdm(
            range(cfg["Training"]["Epochs"]),
            total=int(cfg["Training"]["Epochs"]),
        )
        for epoch in epoch_bar:
            model.train()
            with tqdm(
                train_loader,
                total=len(train_data),
                desc=f"Training",
                leave=False,
            ) as pbar:
                correct = 0
                for batch in train_loader:
                    inputs, targets = batch["input"].to(device), batch[
                        "target"
                    ].to(device)
                    outputs = model(inputs)
                    predicted_outputs = outputs.argmax(dim=1)
                    correct += (predicted_outputs == targets).sum().item()
                    loss = criterion(outputs.squeeze(1), targets)
                    # tensorboard_writer_train.add_scalar("Loss", loss.item(), global_step)
                    pbar.set_postfix(**{"loss": loss.item()})
                    loss.backward()
                    optimizer.step()
                    pbar.update(inputs.shape[0])
                    accuracy = metrics.accuracy_score(
                        targets.detach().cpu().numpy().tolist(),
                        outputs.argmax(dim=1).detach().cpu().numpy().tolist(),
                    )
                    f1 = metrics.f1_score(
                        targets.detach().cpu().numpy().tolist(),
                        outputs.argmax(dim=1).detach().cpu().numpy().tolist(),
                        average="micro",
                    )
                    # tensorboard_writer_train.add_scalar("metrics/F1", f1, global_step)
                    # tensorboard_writer_train.add_scalar(
                    #     "metrics/Accuracy", accuracy, global_step
                    # )
                    logging.info(
                        f"======= Step: {global_step} - Epoch: {epoch+1} - Train Loss: {loss} - Accuracy: {accuracy} - F1 score: {f1} ======="
                    )
                    global_step += 1
                    if (
                        global_step
                        % (len(train_data) // (1 * cfg["Data"]["Batch_size"]))
                        == 0
                    ):
                        valid_loss, valid_metrics = evaluate(
                            model,
                            valid_loader,
                            criterion,
                            optimizer,
                            device,
                            classes,
                        )
                        results.append(valid_metrics["Accuracy"])

                        logging.info(
                            f'======= Step: {global_step} - Epoch: {epoch+1} - Valid Loss: {valid_loss} - F1:{valid_metrics["F1"]} - Accuracy:{valid_metrics["Accuracy"]}======='
                        )
                        for class_name, acc in valid_metrics[
                            "Per_class_accuracy"
                        ].items():
                            # tensorboard_writer_valid.add_scalar(f"Per_Class_Accuracy/{class_name}", acc, global_step)
                            logging.info(
                                f"Accuracy for {class_name}: {acc*100:.2f}%"
                            )

                        # tensorboard_writer_valid.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                        # tensorboard_writer_valid.add_scalar("Loss", valid_loss, global_step)
                        # tensorboard_writer_valid.add_scalar(
                        #     "metrics/F1", valid_metrics["F1"], global_step
                        # )
                        # tensorboard_writer_valid.add_scalar(
                        #     "metrics/Accuracy", valid_metrics["Accuracy"], global_step
                        # )
                        # Save metrics to csv :
                        metrics_data = {
                            "Step": global_step,
                            "Epoch": epoch + 1,
                            "Valid Loss": valid_loss,
                            "Overall Accuracy": valid_metrics["Accuracy"],
                            "F1 Score": valid_metrics["F1"],
                        }
                        metrics_data.update(
                            {
                                f"Accuracy_{class_name}": acc
                                for class_name, acc in valid_metrics[
                                    "Per_class_accuracy"
                                ].items()
                            }
                        )
                        csv_file_path = os.path.join(logdir, "accuracies.csv")
                        save_metrics_to_csv(metrics_data, csv_file_path)

                        all_true_labels.extend(
                            valid_metrics["CM"].index.tolist()
                        )
                        all_predicted_labels.extend(
                            valid_metrics["CM"].columns.tolist()
                        )
                        valid_metrics["CM"].to_csv(f"CM_.csv", index=False)
                        if valid_metrics["Accuracy"] > best_results:
                            torch.save(model.state_dict(), best_model_path)
                            best_results = valid_metrics["Accuracy"]
                            logging.info(
                                f"=== Best model saved with Accuracy score: {valid_metrics['Accuracy']} | Best F1-Score: {valid_metrics['F1']} ===="
                            )

            epoch_bar.set_description(
                f'Epoch [{epoch+1}/{cfg["Training"]["Epochs"]}]'
            )
            epoch_bar.refresh()

        cf_matrix = metrics.confusion_matrix(
            all_true_labels, all_predicted_labels
        )
        df_cm = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
            index=classes,
            columns=classes,
        )
        df_cm.to_csv("confusion_matrix.csv", index=True)
        # tensorboard_writer_valid.close()
        # tensorboard_writer_train.close()
    max_score = max(results)
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    print("--------------------------------")
    print(f"Average: {sum(results)/len(results) * 100} %")
    print(f"Max: {max_score} --- FOLD:{results.index(max_score)}")


if __name__ == "__main__":
    args = parse()
    main(args)
