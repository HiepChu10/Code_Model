import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from utils import clear_memory, check_memory_threshold


class ModelEvaluator:
    def __init__(self, output_dir, memory_settings=None):
        self.output_dir = output_dir
        self.memory_settings = memory_settings or {}
        
    def evaluate_and_plot(self, model, X_data, y_true, threshold, name_prefix, title_suffix, log_print):
        """Evaluate model and create plots"""
        y_proba = model.predict_proba(X_data)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Use standard accuracy for fair comparison (threshold 0.5)
        acc = model.score(X_data, y_true)
        
        # Get classification report as dictionary
        report_dict = classification_report(y_true, y_pred, output_dict=True)  # type: ignore
        
        log_print(f"\nBáo cáo phân loại ({title_suffix}):")
        for label_name, metrics in report_dict.items():  # type: ignore
            if isinstance(metrics, dict):
                log_print(f"{label_name:<12} Precision: {metrics['precision']:.2f}  Recall: {metrics['recall']:.2f}  F1-score: {metrics['f1-score']:.2f}  Support: {int(metrics['support'])}")
            elif label_name == 'accuracy':
                log_print(f"Accuracy: {metrics:.2f}")

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(self.output_dir, f"report_{name_prefix}.csv"))

        # Create prediction bar chart
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_1 = tp + fp
        prec_1 = tp / total_1 if total_1 > 0 else 0

        labels = ['Tổng dự đoán nhãn 1', 'Dự đoán đúng (TP)', 'Dự đoán sai (FP)']
        values = [total_1, tp, fp]
        colors = ['blue', 'green', 'red']

        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, values, color=colors)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01*yval,
                     f'{int(yval)}', ha='center', va='bottom')
        plt.title(f'Precision nhãn 1 = {prec_1:.2f} - {title_suffix}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"prediction_bar_{name_prefix}.png"))
        plt.close()

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.title(f'Confusion Matrix - {title_suffix}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{name_prefix}.png"))
        plt.close()

        return report_dict, acc
    
    def save_metrics_to_csv(self, report_dict, y_type, threshold, feature_set_idx, log_print=None):
        """
        Save precision and recall metrics for labels 0 and 1 to CSV file
        """
        try:
            # Lưu CSV vào folder autorun (cố định)
            csv_path = os.path.join(os.path.dirname(__file__), "metrics_log.csv")
            
            # Extract metrics for labels 0 and 1
            metrics_0 = report_dict.get('0', {})
            metrics_1 = report_dict.get('1', {})
            
            # Prepare row data - format thành phần trăm
            row_data = {
                'label_0_precision': f"{metrics_0.get('precision', 0) * 100:.2f}%",
                'label_0_recall': f"{metrics_0.get('recall', 0) * 100:.2f}%",
                'label_1_precision': f"{metrics_1.get('precision', 0) * 100:.2f}%",
                'label_1_recall': f"{metrics_1.get('recall', 0) * 100:.2f}%"
            }
            
            # Create DataFrame for new row
            new_row_df = pd.DataFrame([row_data])
            
            # Check if CSV file exists
            if os.path.exists(csv_path):
                # File exists, append to it
                new_row_df.to_csv(csv_path, mode='a', header=False, index=False)
                if log_print:
                    log_print(f"📊 Metrics appended to {csv_path}")
            else:
                # File doesn't exist, create it with header
                new_row_df.to_csv(csv_path, mode='w', header=True, index=False)
                if log_print:
                    log_print(f"📊 Metrics CSV created: {csv_path}")
                    
        except Exception as e:
            if log_print:
                log_print(f"⚠️  Error saving metrics to CSV: {str(e)}")
    
    def plot_feature_importance(self, model, features, log_print=None):
        """Plot feature importance with memory optimization"""
        try:
            if log_print:
                log_print("📊 Creating feature importance plot...")
            
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(importances)), importances[sorted_idx], color='skyblue')
            plt.yticks(range(len(importances)), [features[i] for i in sorted_idx])
            plt.xlabel("Feature Importance")
            plt.title("Tầm quan trọng của đặc trưng")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
            plt.close()  # Important: close to free memory
            
        except Exception as e:
            if log_print:
                log_print(f"⚠️  Error creating feature importance plot: {str(e)}")