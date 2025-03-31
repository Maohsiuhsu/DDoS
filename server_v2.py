import os
import flwr as fl
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json

# 設定儲存路徑
save_model_dir = "./2017/ADASYN"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_model_dir = os.path.join(save_model_dir, current_time)
os.makedirs(save_model_dir, exist_ok=True)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_available_clients=2):
        super().__init__(min_available_clients=min_available_clients)
        self.client_training_data = []
        self.client_evaluation_data = []
        self.client_mapping = {}  # 存 client id 對應的名稱
        self.client_counter = 1  # 記錄目前的 client 數量

        self.losses_distributed = []
        self.metrics_distributed = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1-score": []
        }
        self.losses_centralized = []
        self.metrics_centralized = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1-score": []
        }
        # 新增屬性，用來存放未知的 additional metrics
        self.additional_metrics_data = []
        self.additional_metrics_data_eval = []

    def get_client_id(self, original_id):
        """如果是新的 client，就分配一個新的 `client_X` ID"""
        if original_id not in self.client_mapping:
            new_client_id = f"client_{self.client_counter}"
            self.client_mapping[original_id] = new_client_id
            self.client_counter += 1
        return self.client_mapping[original_id]

    def aggregate_fit(self, server_round, results, failures):
        print(f"Round {server_round}: Received training results")

        round_train_losses = []
        round_train_metrics = {key: [] for key in self.metrics_distributed.keys()}
        client_data = []

        for client, fit_res in results:
            client_id = self.get_client_id(client.cid)  # 取得對應的 client_ 編號
            metrics = fit_res.metrics

            # 取得固定的 metrics
            loss = metrics.get("loss", 0)
            accuracy = metrics.get("accuracy", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1-score", 0)

            round_train_losses.append(loss)
            round_train_metrics["accuracy"].append(accuracy)
            round_train_metrics["precision"].append(precision)
            round_train_metrics["recall"].append(recall)
            round_train_metrics["f1-score"].append(f1_score)

            client_data.append([server_round, client_id, loss, accuracy, precision, recall, f1_score])
            
            # 取得額外的 metrics（未知的 key）
            additional_metrics = {key: value for key, value in metrics.items() 
                                  if key not in ["loss", "accuracy", "precision", "recall", "f1-score"]}
            if additional_metrics:
                for key, value in additional_metrics.items():
                    self.additional_metrics_data.append([server_round, client_id, key, value])

        self.client_training_data.extend(client_data)

        avg_train_loss = np.mean(round_train_losses) if round_train_losses else 0
        avg_train_metrics = {key: np.mean(values) if values else 0 for key, values in round_train_metrics.items()}

        print(f"Round {server_round}: Avg Train Loss: {avg_train_loss:.4f}")
        print(f"Round {server_round}: Accuracy: {avg_train_metrics['accuracy']:.4f}, "
              f"Precision: {avg_train_metrics['precision']:.4f}, "
              f"Recall: {avg_train_metrics['recall']:.4f}, "
              f"F1-Score: {avg_train_metrics['f1-score']:.4f}")

        self.losses_distributed.append((server_round, avg_train_loss))
        for key in self.metrics_distributed.keys():
            self.metrics_distributed[key].append((server_round, avg_train_metrics[key]))

        # 聚合權重
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        # 儲存 Aggregated Model Weights 與 Metrics
        if aggregated_weights is not None:
            round_dir = os.path.join(save_model_dir, "weights")
            os.makedirs(round_dir, exist_ok=True)
            weight_file = os.path.join(round_dir, f"weights_{server_round}.npz")
            np.savez(weight_file, *aggregated_weights)
            print(f"Saved aggregated model weights for Round {server_round} at {weight_file}")

            round_metrics_dir = os.path.join(round_dir, "metrics")
            os.makedirs(round_metrics_dir, exist_ok=True)
            metrics_file = os.path.join(round_metrics_dir, f"metrics_{server_round}.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Round: {server_round}\n")
                f.write(f"Average Loss: {avg_train_loss:.4f}\n")
                f.write(f"Average Accuracy: {avg_train_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {avg_train_metrics['precision']:.4f}\n")
                f.write(f"Recall: {avg_train_metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {avg_train_metrics['f1-score']:.4f}\n")
            print(f"Saved metrics for Round {server_round} at {metrics_file}")

        return aggregated_weights

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"Round {server_round}: Received evaluation results")

        round_eval_losses = []
        round_eval_metrics = {key: [] for key in self.metrics_centralized.keys()}
        client_data = []

        for client, eval_res in results:
            client_id = self.get_client_id(client.cid)

            metrics = eval_res.metrics
            loss = eval_res.loss
            accuracy = eval_res.metrics.get("accuracy", 0)
            precision = eval_res.metrics.get("precision", 0)
            recall = eval_res.metrics.get("recall", 0)
            f1_score = eval_res.metrics.get("f1-score", 0)

            round_eval_losses.append(loss)
            round_eval_metrics["accuracy"].append(accuracy)
            round_eval_metrics["precision"].append(precision)
            round_eval_metrics["recall"].append(recall)
            round_eval_metrics["f1-score"].append(f1_score)


            # 取得額外的 metrics（未知的 key）
            additional_metrics = {key: value for key, value in metrics.items() 
                                  if key not in ["loss", "accuracy", "precision", "recall", "f1-score"]}
            if additional_metrics:
                for key, value in additional_metrics.items():
                    self.additional_metrics_data_eval.append([server_round, client_id, key, value])


            client_data.append([server_round, client_id, loss, accuracy, precision, recall, f1_score])

        self.client_evaluation_data.extend(client_data)

        avg_eval_loss = np.mean(round_eval_losses) if round_eval_losses else 0
        avg_eval_metrics = {key: np.mean(values) if values else 0 for key, values in round_eval_metrics.items()}

        print(f"Round {server_round}: Avg Eval Loss: {avg_eval_loss:.4f}")
        print(f"Round {server_round}: Accuracy: {avg_eval_metrics['accuracy']:.4f}, "
              f"Precision: {avg_eval_metrics['precision']:.4f}, "
              f"Recall: {avg_eval_metrics['recall']:.4f}, "
              f"F1-Score: {avg_eval_metrics['f1-score']:.4f}")

        self.losses_centralized.append((server_round, avg_eval_loss))
        for key in self.metrics_centralized.keys():
            self.metrics_centralized[key].append((server_round, avg_eval_metrics[key]))

        return avg_eval_loss, avg_eval_metrics

    def save_results(self):
        """儲存 Client 資料以及所有 Client 的平均數據"""
        df_train = pd.DataFrame(self.client_training_data, columns=["Round", "Client_ID", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])
        df_eval = pd.DataFrame(self.client_evaluation_data, columns=["Round", "Client_ID", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])

        df_train.to_csv(os.path.join(save_model_dir, "client_training_data.csv"), index=False)
        df_eval.to_csv(os.path.join(save_model_dir, "client_evaluation_data.csv"), index=False)

        avg_train_metrics = df_train.groupby("Round")[["Loss", "Accuracy", "Precision", "Recall", "F1-Score"]].mean().reset_index()
        avg_eval_metrics = df_eval.groupby("Round")[["Loss", "Accuracy", "Precision", "Recall", "F1-Score"]].mean().reset_index()

        avg_train_metrics.to_csv(os.path.join(save_model_dir, "average_training_data.csv"), index=False)
        avg_eval_metrics.to_csv(os.path.join(save_model_dir, "average_evaluation_data.csv"), index=False)

        # 儲存 client mapping
        with open(os.path.join(save_model_dir, "client_mapping.json"), "w") as f:
            json.dump(self.client_mapping, f, indent=4)

        # 儲存 additional metrics（未知 key）
        if self.additional_metrics_data:
            df_additional = pd.DataFrame(self.additional_metrics_data, columns=["Round", "Client_ID", "Metric", "Value"])
            additional_metrics_file = os.path.join(save_model_dir, "additional_metrics.csv")
            df_additional.to_csv(additional_metrics_file, index=False)
            print(f"Saved additional metrics to {additional_metrics_file}")

        print("Training and evaluation data saved as CSV and JSON.")

    def visualize_progress(self):
        """畫出每個 Client 與所有 Client 的訓練與評估數據"""
        df_train = pd.DataFrame(self.client_training_data, columns=["Round", "Client_ID", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])
        df_eval = pd.DataFrame(self.client_evaluation_data, columns=["Round", "Client_ID", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])

        # 每個 Client 的 Metrics 圖
        for client_id in df_train["Client_ID"].unique():
            client_data_train = df_train[df_train["Client_ID"] == client_id]
            client_data_eval = df_eval[df_eval["Client_ID"] == client_id]

            plt.figure(figsize=(12, 6))
            plt.plot(client_data_train["Round"], client_data_train["Accuracy"], marker='o', label="Train Accuracy", linestyle='-')
            plt.plot(client_data_train["Round"], client_data_train["Precision"], marker='s', label="Train Precision", linestyle='-')
            plt.plot(client_data_train["Round"], client_data_train["Recall"], marker='^', label="Train Recall", linestyle='-')
            plt.plot(client_data_train["Round"], client_data_train["F1-Score"], marker='x', label="Train F1-Score", linestyle='-')

            plt.plot(client_data_eval["Round"], client_data_eval["Accuracy"], marker='o', label="Eval Accuracy", linestyle='--')
            plt.plot(client_data_eval["Round"], client_data_eval["Precision"], marker='s', label="Eval Precision", linestyle='--')
            plt.plot(client_data_eval["Round"], client_data_eval["Recall"], marker='^', label="Eval Recall", linestyle='--')
            plt.plot(client_data_eval["Round"], client_data_eval["F1-Score"], marker='x', label="Eval F1-Score", linestyle='--')

            plt.title(f"{client_id} Metrics per Round (Train & Eval)")
            plt.xlabel("Training Round")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plot_file = os.path.join(save_model_dir, f"{client_id}_metrics.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved {client_id} metrics plot at {plot_file}")

        # 所有 Client 平均 Metrics
        avg_train_metrics = df_train.groupby("Round")[["Accuracy", "Precision", "Recall", "F1-Score"]].mean()
        avg_eval_metrics = df_eval.groupby("Round")[["Accuracy", "Precision", "Recall", "F1-Score"]].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_train_metrics.index, avg_train_metrics["Accuracy"], marker='o', label="Avg Train Accuracy", linestyle='-')
        plt.plot(avg_train_metrics.index, avg_train_metrics["Precision"], marker='s', label="Avg Train Precision", linestyle='-')
        plt.plot(avg_train_metrics.index, avg_train_metrics["Recall"], marker='^', label="Avg Train Recall", linestyle='-')
        plt.plot(avg_train_metrics.index, avg_train_metrics["F1-Score"], marker='x', label="Avg Train F1-Score", linestyle='-')

        plt.plot(avg_eval_metrics.index, avg_eval_metrics["Accuracy"], marker='o', label="Avg Eval Accuracy", linestyle='--')
        plt.plot(avg_eval_metrics.index, avg_eval_metrics["Precision"], marker='s', label="Avg Eval Precision", linestyle='--')
        plt.plot(avg_eval_metrics.index, avg_eval_metrics["Recall"], marker='^', label="Avg Eval Recall", linestyle='--')
        plt.plot(avg_eval_metrics.index, avg_eval_metrics["F1-Score"], marker='x', label="Avg Eval F1-Score", linestyle='--')

        plt.title("Average Metrics per Round (All Clients)")
        plt.xlabel("Training Round")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        avg_plot_file = os.path.join(save_model_dir, "average_metrics.png")
        plt.savefig(avg_plot_file)
        plt.close()
        print(f"Saved average metrics plot at {avg_plot_file}")

        # 每個 Client 的 Loss 圖
        for client_id in df_train["Client_ID"].unique():
            client_data_train = df_train[df_train["Client_ID"] == client_id]
            client_data_eval = df_eval[df_eval["Client_ID"] == client_id]

            plt.figure(figsize=(12, 6))
            plt.plot(client_data_train["Round"], client_data_train["Loss"], marker='o', label="Train Loss", linestyle='-')
            plt.plot(client_data_eval["Round"], client_data_eval["Loss"], marker='o', label="Eval Loss", linestyle='--')

            plt.title(f"{client_id} Loss per Round (Train & Eval)")
            plt.xlabel("Training Round")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plot_file = os.path.join(save_model_dir, f"{client_id}_loss.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved {client_id} loss plot at {plot_file}")

        # 所有 Client 平均 Loss 圖
        avg_train_loss = df_train.groupby("Round")["Loss"].mean()
        avg_eval_loss = df_eval.groupby("Round")["Loss"].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_train_loss.index, avg_train_loss, marker='o', label="Avg Train Loss", linestyle='-')
        plt.plot(avg_eval_loss.index, avg_eval_loss, marker='o', label="Avg Eval Loss", linestyle='--')

        plt.title("Average Loss per Round (All Clients)")
        plt.xlabel("Training Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        avg_loss_file = os.path.join(save_model_dir, "average_loss.png")
        plt.savefig(avg_loss_file)
        plt.close()
        print(f"Saved average loss plot at {avg_loss_file}")

        # # 繪製未知 metric 的圖表 (訓練)
        # if self.additional_metrics_data:
        #     df_additional = pd.DataFrame(self.additional_metrics_data, columns=["Round", "Client_ID", "Metric", "Value"])
        #     for client_id in df_additional["Client_ID"].unique():
        #         client_data = df_additional[df_additional["Client_ID"] == client_id]
        #         plt.figure(figsize=(12, 6))
        #         for metric in client_data["Metric"].unique():
        #             metric_data = client_data[client_data["Metric"] == metric]
        #             plt.plot(metric_data["Round"], metric_data["Value"], marker='o', label=f"{metric}")
        #         plt.title(f"Additional Metrics per Round for {client_id}")
        #         plt.xlabel("Training Round")
        #         plt.ylabel("Metric Value")
        #         plt.legend()
        #         plt.grid(True)
        #         plot_file = os.path.join(save_model_dir, f"{client_id}_additional_metrics.png")
        #         plt.savefig(plot_file)
        #         plt.close()
        #         print(f"Saved additional metrics plot for {client_id} at {plot_file}")

        # # 繪製未知 metric 的圖表 (評估)
        # if self.additional_metrics_data_eval:
        #     df_additional_eval = pd.DataFrame(self.additional_metrics_data_eval, columns=["Round", "Client_ID", "Metric", "Value"])
        #     for client_id in df_additional_eval["Client_ID"].unique():
        #         client_data = df_additional_eval[df_additional_eval["Client_ID"] == client_id]
        #         plt.figure(figsize=(12, 6))
        #         for metric in client_data["Metric"].unique():
        #             metric_data = client_data[client_data["Metric"] == metric]
        #             plt.plot(metric_data["Round"], metric_data["Value"], marker='o', label=f"{metric}")
        #         plt.title(f"Additional Evaluation Metrics per Round for {client_id}")
        #         plt.xlabel("Training Round")
        #         plt.ylabel("Metric Value")
        #         plt.legend()
        #         plt.grid(True)
        #         plot_file = os.path.join(save_model_dir, f"{client_id}_additional_metrics_eval.png")
        #         plt.savefig(plot_file)
        #         plt.close()
        #         print(f"Saved additional evaluation metrics plot for {client_id} at {plot_file}")

        # 繪製未知 metric 的圖表 (訓練) - 根據 precision, recall, f1-score 分開產生不同圖片
        if self.additional_metrics_data:
            df_additional = pd.DataFrame(self.additional_metrics_data, columns=["Round", "Client_ID", "Metric", "Value"])
            for client_id in df_additional["Client_ID"].unique():
                client_data = df_additional[df_additional["Client_ID"] == client_id]
                # 針對每個 metric 類型分別處理
                for metric_type in ["precision", "recall", "f1-score"]:
                    # 篩選出 Metric 欄位中包含該 metric_type 字串的資料 (忽略大小寫)
                    metric_data = client_data[client_data["Metric"].str.contains(metric_type, case=False)]
                    if metric_data.empty:
                        continue  # 若沒有資料則跳過
                    plt.figure(figsize=(12, 6))
                    # 如果有多個不同的 label (例如 "class1_precision", "class2_precision")，就分別繪圖
                    for label in metric_data["Metric"].unique():
                        label_data = metric_data[metric_data["Metric"] == label]
                        plt.plot(label_data["Round"], label_data["Value"], marker='o', label=label)
                    plt.title(f"{metric_type.capitalize()} Metrics per Round for {client_id} (Training)")
                    plt.xlabel("Training Round")
                    plt.ylabel("Metric Value")
                    plt.legend()
                    plt.grid(True)
                    plot_file = os.path.join(save_model_dir, f"{client_id}_{metric_type}_additional_metrics.png")
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Saved {metric_type} additional metrics plot for {client_id} at {plot_file}")

        # 繪製未知 metric 的圖表 (評估) - 根據 precision, recall, f1-score 分開產生不同圖片
        if self.additional_metrics_data_eval:
            df_additional_eval = pd.DataFrame(self.additional_metrics_data_eval, columns=["Round", "Client_ID", "Metric", "Value"])
            for client_id in df_additional_eval["Client_ID"].unique():
                client_data = df_additional_eval[df_additional_eval["Client_ID"] == client_id]
                for metric_type in ["precision", "recall", "f1-score"]:
                    metric_data = client_data[client_data["Metric"].str.contains(metric_type, case=False)]
                    if metric_data.empty:
                        continue
                    plt.figure(figsize=(12, 6))
                    for label in metric_data["Metric"].unique():
                        label_data = metric_data[metric_data["Metric"] == label]
                        plt.plot(label_data["Round"], label_data["Value"], marker='o', label=label)
                    plt.title(f"{metric_type.capitalize()} Evaluation Metrics per Round for {client_id}")
                    plt.xlabel("Training Round")
                    plt.ylabel("Metric Value")
                    plt.legend()
                    plt.grid(True)
                    plot_file = os.path.join(save_model_dir, f"{client_id}_{metric_type}_additional_metrics_eval.png")
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Saved {metric_type} additional evaluation metrics plot for {client_id} at {plot_file}")



# 建立策略物件並啟動伺服器
strategy = SaveModelStrategy(min_available_clients=2)

history = fl.server.start_server(
    server_address="120.113.101.8:8081", 
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

strategy.save_results()
strategy.visualize_progress()
