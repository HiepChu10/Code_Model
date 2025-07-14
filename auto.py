import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches


# ===================== CONFIG =====================
DATA_PATH = 'F:/dataML/data/TRX/TRX_2025-03-01_00-00-00.csv'
TEST_NEW_PATH = 'F:/dataML/data/TRX/TRX_2025-06-30_02-59-12.csv'

settings = [
    {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.8},
    {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.8},
    {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.9},
    {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.9},
    {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.8},
    {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.8},
    {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.9},
    {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.9}
]

feature_sets = [
    [
        'sma_macd_diff_5', 'sma_macd_diff_10', 'sma_macd_diff_sub', 'macd_diff', 'macd',
        'close_div_ema9', 'close_div_ema25', 'close_div_ema50',
        'close_div_sma9', 'close_div_sma25', 'close_div_sma50',
        'close_div_ema100', 'close_div_ema200', 'close_div_sma100', 'close_div_sma200',
        'volume_usdt', 'volume_usdt_10', 'volume_usdt_60'
    ],
    [
        'sma_macd_diff_5', 'sma_macd_diff_10', 'sma_macd_diff_sub', 'macd_diff', 'macd',
        'close_div_ema9', 'close_div_ema25', 'close_div_ema50',
        'close_div_sma9', 'close_div_sma25', 'close_div_sma50',
        'close_div_ema100', 'close_div_ema200', 'close_div_sma100', 'close_div_sma200'
    ]
]

# Global variables to store results for Word report
global_results = {}

for feat_idx, features in enumerate(feature_sets):
    for config in settings:
        Y_TYPE = config['Y_TYPE']
        threshold = config['threshold']
        label = f"{Y_TYPE}_thr{threshold}"
        output_dir = os.path.join("output", label, f"featureset_{feat_idx+1}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n\n========== 🔁 Running: {label} | FeatureSet {feat_idx+1} ==========")
        log_file = open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8")

        def log_print(msg):
            print(msg)
            log_file.write(str(msg) + "\n")

        df = pd.read_csv(DATA_PATH)
        df['OpenTime'] = pd.to_datetime(df['OpenTime'])
        df.set_index('OpenTime', inplace=True)

        log_print(f"Số lượng giá trị 0: {(df[Y_TYPE]==0).sum()}")
        log_print(f"Số lượng giá trị 1: {(df[Y_TYPE]==1).sum()}")
        log_print(f"Tổng số bản ghi: {len(df)}")

        X = df[features]
        y = df[Y_TYPE]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, random_state=42
        )

        train_dist = y_train.value_counts(normalize=True) * 100
        log_print("\nTỷ lệ nhãn tập train:")
        log_print(train_dist.round(2).to_string())

        # Fix: Proper calculation of scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=20
        )

        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        log_print(f"\n⏱️ Thời gian huấn luyện (warm_start): {elapsed:.2f} giây")

        # Store results for Word report
        global_results[label + f"_fs{feat_idx+1}"] = {
            'Y_TYPE': Y_TYPE,
            'threshold': threshold,
            'features': features,
            'elapsed': elapsed,
            'scale_pos_weight': scale_pos_weight,
            'y_train': y_train,
            'y_test': y_test,
            'X_train': X_train,
            'X_test': X_test,
            'xgb_model': xgb_model,
            'output_dir': output_dir
        }

        def evaluate_and_plot(X_data, y_true, name_prefix, title_suffix):
            y_proba = xgb_model.predict_proba(X_data)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            
            # Get classification report as dictionary
            report_dict = classification_report(y_true, y_pred, output_dict=True)  # type: ignore
            
            log_print(f"\n📊 Báo cáo phân loại ({title_suffix}):")
            for label_name, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    log_print(f"{label_name:<12} Precision: {metrics['precision']:.2f}  Recall: {metrics['recall']:.2f}  F1-score: {metrics['f1-score']:.2f}  Support: {int(metrics['support'])}")
                elif label_name == 'accuracy':
                    log_print(f"Accuracy: {metrics:.2f}")

            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(os.path.join(output_dir, f"report_{name_prefix}.csv"))

            # Store detailed metrics for Word report
            global_results[label + f"_fs{feat_idx+1}"][f'report_{name_prefix}'] = report_dict
            global_results[label + f"_fs{feat_idx+1}"][f'accuracy_{name_prefix}'] = acc

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
            plt.savefig(os.path.join(output_dir, f"prediction_bar_{name_prefix}.png"))
            plt.close()

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                        xticklabels=['0', '1'], yticklabels=['0', '1'])
            plt.title(f'Confusion Matrix - {title_suffix}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name_prefix}.png"))
            plt.close()

            return report_dict, acc

        # Evaluate on test set
        test_report, test_acc = evaluate_and_plot(X_test, y_test, 'test', 'Tập Test')

        # Feature importance
        importances = xgb_model.feature_importances_
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(importances)), importances[sorted_idx], color='skyblue')
        plt.yticks(range(len(importances)), [features[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Tầm quan trọng của đặc trưng")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance.png"))
        plt.close()

        # Evaluate on new data
        df_new = pd.read_csv(TEST_NEW_PATH)
        count_rows_new = len(df_new)
        count_0 = (df_new[Y_TYPE] == 0).sum()
        count_1 = (df_new[Y_TYPE] == 1).sum()
        pct_0 = count_0 / count_rows_new * 100
        pct_1 = count_1 / count_rows_new * 100
        log_print(f"\n📦 Dữ liệu mới: {count_rows_new} bản ghi (0: {count_0} - {pct_0:.2f}%, 1: {count_1} - {pct_1:.2f}%)")

        # Store new data info for Word report
        global_results[label + f"_fs{feat_idx+1}"]["new_data"] = {
            'count_rows_new': count_rows_new,
            'count_0': count_0,
            'count_1': count_1,
            'pct_0': pct_0,
            'pct_1': pct_1
        }

        new_report = None
        new_acc = None
        if Y_TYPE in df_new.columns:
            y_new = df_new[Y_TYPE]
            X_new = df_new[features]
            new_report, new_acc = evaluate_and_plot(X_new, y_new, 'new', 'Tập Mới')
        else:
            log_print("Không tìm thấy nhãn trong dữ liệu mới.")

        log_print(f"\n✅ Đã lưu tất cả kết quả vào: {output_dir}")
        log_file.close()

        def format_int(n):
            return f"{int(n):,}" if pd.notna(n) and n is not None else "-"

        def format_percentage(n):
            return f"{n:.2%}" if pd.notna(n) and n is not None else "-"

        def generate_report_word(report_counter=1):
            doc = Document()
            doc.add_heading("REPORT", level=1)
            doc.add_paragraph("")

            # Get current results
            current_key = label + f"_fs{feat_idx+1}"
            results = global_results[current_key]

            # ------------------ MỤC 1 ------------------
            doc.add_heading("1. Thông tin tổng quan", level=2)

            # Get token name from file path
            token_name = os.path.basename(DATA_PATH).split("_")[0].upper()

            info_1 = [
                ("Tên mô hình", "XGBoost"),
                ("Pair token", token_name),
                ("Dự đoán MinLow/MaxHigh", "MinLow" if "minlow" in Y_TYPE else "MaxHigh"),
                ("Threashold", str(threshold)),
                ("Rate", "0.1" if "r10" in Y_TYPE else "0.2" if "r20" in Y_TYPE else "0.08"),
                ("Time frame prediction", "120M"),
                ("Có cân bằng dữ liệu không", "Có"),
            ]
            table_1 = doc.add_table(rows=1, cols=2)
            table_1.style = 'Table Grid'
            table_1.rows[0].cells[0].text = "Thông tin"
            table_1.rows[0].cells[1].text = "Giá trị"
            for label_text, value in info_1:
                row = table_1.add_row().cells
                row[0].text = label_text
                row[1].text = value

            # ------------------ MỤC 1.2: Thống kê tập train ------------------
            doc.add_paragraph("\nTập dữ liệu train", style='Heading 3')
            
            # Get correct counts - full dataset first, then train set
            df_full = pd.read_csv(DATA_PATH)
            label_0_full = (df_full[Y_TYPE] == 0).sum()
            label_1_full = (df_full[Y_TYPE] == 1).sum()
            total_full = len(df_full)
            
            # Train set counts
            label_0_train = (results['y_train'] == 0).sum()
            label_1_train = (results['y_train'] == 1).sum()
            total_train = len(results['y_train'])
            pct_1_train = label_1_train / total_train * 100
            pct_0_train = label_0_train / total_train * 100
            
            train_info = [
                ("Tên bộ dữ liệu", os.path.basename(DATA_PATH)),
                ("Tỉ lệ dữ liệu train/ test", "70 / 30"),
                ("Các thuộc tính đầu vào", "\n".join([f"'{f}'" for f in features])),
                ("Số lượng nhãn 0", format_int(label_0_full)),
                ("Số lượng nhãn 1", format_int(label_1_full)),
                ("Số lượng bản ghi", format_int(total_full)),
                ("Số lượng nhãn 1 tập train", format_int(label_1_train)),
                ("Tỉ lệ % nhãn 1 tập train", f"{pct_1_train:.1f}%"),
                ("Số lượng nhãn 0 tập train", format_int(label_0_train)),
                ("Tỉ lệ % nhãn 0 tập train", f"{pct_0_train:.1f}%")
            ]
            table_train = doc.add_table(rows=1, cols=2)
            table_train.style = 'Table Grid'
            table_train.rows[0].cells[0].text = "Thông tin"
            table_train.rows[0].cells[1].text = "Giá trị"
            for label_text, value in train_info:
                row = table_train.add_row().cells
                row[0].text = label_text
                row[1].text = value

            # ------------------ MỤC 1.3: Dữ liệu mới ------------------
            doc.add_paragraph("\nTập dữ liệu Test (nằm ngoài tập dữ liệu train)", style='Heading 3')
            
            new_data_info = results["new_data"]
            test_info = [
                ("Tên bộ dữ liệu tập test", os.path.basename(TEST_NEW_PATH)),
                ("Số lượng bản ghi", format_int(new_data_info['count_rows_new'])),
                ("Số lượng nhãn 0 tập test", format_int(new_data_info['count_0'])),
                ("Tỉ lệ % nhãn 0 tập test", f"{new_data_info['pct_0']:.2f}%"),
                ("Số lượng nhãn 1 tập test", format_int(new_data_info['count_1'])),
                ("Tỉ lệ % nhãn 1 tập test", f"{new_data_info['pct_1']:.2f}%")
            ]
            table_test = doc.add_table(rows=1, cols=2)
            table_test.style = 'Table Grid'
            table_test.rows[0].cells[0].text = "Thông tin"
            table_test.rows[0].cells[1].text = "Giá trị"
            for label_text, value in test_info:
                row = table_test.add_row().cells
                row[0].text = label_text
                row[1].text = value

            # ------------------ MỤC 2 ------------------
            doc.add_heading("2. Bộ tham số đầu vào", level=2)
            param_str = f"""
                xgb_model = XGBClassifier(
     n_estimators=300,           # Số lượng cây
    learning_rate=0.05,         # Tốc độ học
    max_depth=8,                # Độ sâu tối đa của cây
    min_child_weight=2,         # Trọng số tối thiểu của child node
    subsample=0.8,              # Tỷ lệ mẫu cho mỗi cây
    colsample_bytree=0.8,       # Tỷ lệ features cho mỗi cây
    reg_alpha=0.01,             # L1 regularization
    reg_lambda=0.1,             # L2 regularization
    scale_pos_weight={results['scale_pos_weight']:.2f},  # Xử lý imbalanced data
                    random_state=42,
    eval_metric='logloss',      # Metric để đánh giá
                    verbosity=0,
                    n_jobs=20
                )
            """.strip()
            doc.add_paragraph(param_str)

            # ------------------ MỤC 3 ------------------
            doc.add_heading("3. Kết quả chạy model", level=2)
            
            # Calculate accuracies using stored results
            train_accuracy = results['xgb_model'].score(results['X_train'], results['y_train']) * 100
            test_accuracy = results['accuracy_test'] * 100
            
            # Format as separate paragraphs like in the example
            doc.add_paragraph(f"Thời gian chạy\n\n{results['elapsed']:.1f}s")
            doc.add_paragraph(f"Train accuracy\n\n{train_accuracy:.2f}%")
            doc.add_paragraph(f"Test accuracy\n\n{test_accuracy:.2f}%")

            # Classification Report Test table
            test_report = results['report_test']
            
            table_report = doc.add_table(rows=1, cols=5)
            table_report.style = 'Table Grid'
            hdr_cells = table_report.rows[0].cells
            hdr_cells[0].text = "Label"
            hdr_cells[1].text = "Precision"
            hdr_cells[2].text = "Recall"
            hdr_cells[3].text = "F1-score"
            hdr_cells[4].text = "Support"
            
            # Add rows in specific order: 0, 1, accuracy, macro avg, weighted avg
            rows_data = []
            
            # Add class 0 and 1 first
            for class_label in ['0', '1']:
                if class_label in test_report:
                    metrics = test_report[class_label]
                    rows_data.append([
                        class_label,
                        format_percentage(metrics.get('precision', 0)),
                        format_percentage(metrics.get('recall', 0)),
                        format_percentage(metrics.get('f1-score', 0)),
                        format_int(metrics.get('support', 0))
                    ])
            
            # Add accuracy row
            if 'accuracy' in test_report:
                rows_data.append([
                    'Accuracy',
                    '-',
                    '-',
                    format_percentage(test_report['accuracy']),
                    format_int(len(results['y_test']))
                ])
            
            # Add macro avg
            if 'macro avg' in test_report:
                metrics = test_report['macro avg']
                rows_data.append([
                    'Macro avg',
                    format_percentage(metrics.get('precision', 0)),
                    format_percentage(metrics.get('recall', 0)),
                    format_percentage(metrics.get('f1-score', 0)),
                    format_int(metrics.get('support', 0))
                ])
            
            # Add weighted avg
            if 'weighted avg' in test_report:
                metrics = test_report['weighted avg']
                rows_data.append([
                    'Weighted avg',
                    format_percentage(metrics.get('precision', 0)),
                    format_percentage(metrics.get('recall', 0)),
                    format_percentage(metrics.get('f1-score', 0)),
                    format_int(metrics.get('support', 0))
                ])
            
            # Add all rows to table
            for row_data in rows_data:
                cells = table_report.add_row().cells
                for i, text in enumerate(row_data):
                    cells[i].text = text
            
            # ------------------ MỤC 4 ------------------
            doc.add_heading("4. Các biểu đồ kết quả", level=2)
            for img in ["prediction_bar_test.png", "confusion_matrix_test.png", "feature_importance.png"]:
                path = os.path.join(output_dir, img)
                if os.path.exists(path):
                    doc.add_picture(path, width=Inches(5.5))
                    doc.add_paragraph("")
            
            # ------------------ MỤC 5 & 6: New data results ------------------
            if 'report_new' in results:
                doc.add_heading("5. Kết quả chạy với bộ dữ liệu test ngoài", level=2)
                
                new_report = results['report_new']
                table_new = doc.add_table(rows=1, cols=5)
                table_new.style = 'Table Grid'
                hdr_cells_new = table_new.rows[0].cells
                hdr_cells_new[0].text = "Label"
                hdr_cells_new[1].text = "Precision"
                hdr_cells_new[2].text = "Recall"
                hdr_cells_new[3].text = "F1-score"
                hdr_cells_new[4].text = "Support"
                
                # Same order for new data table
                rows_new_data = []
                
                # Add class 0 and 1 first
                for class_label in ['0', '1']:
                    if class_label in new_report:
                        metrics = new_report[class_label]
                        rows_new_data.append([
                            class_label,
                            format_percentage(metrics.get('precision', 0)),
                            format_percentage(metrics.get('recall', 0)),
                            format_percentage(metrics.get('f1-score', 0)),
                            format_int(metrics.get('support', 0))
                        ])
                
                # Add accuracy row
                if 'accuracy' in new_report:
                    rows_new_data.append([
                        'Accuracy',
                        '-',
                        '-',
                        format_percentage(new_report['accuracy']),
                        format_int(new_data_info['count_rows_new'])
                    ])
                
                # Add macro avg
                if 'macro avg' in new_report:
                    metrics = new_report['macro avg']
                    rows_new_data.append([
                        'Macro avg',
                        format_percentage(metrics.get('precision', 0)),
                        format_percentage(metrics.get('recall', 0)),
                        format_percentage(metrics.get('f1-score', 0)),
                        format_int(metrics.get('support', 0))
                    ])
                
                # Add weighted avg
                if 'weighted avg' in new_report:
                    metrics = new_report['weighted avg']
                    rows_new_data.append([
                        'Weighted avg',
                        format_percentage(metrics.get('precision', 0)),
                        format_percentage(metrics.get('recall', 0)),
                        format_percentage(metrics.get('f1-score', 0)),
                        format_int(metrics.get('support', 0))
                    ])
                
                # Add all rows to new table
                for row_data in rows_new_data:
                    cells = table_new.add_row().cells
                    for i, text in enumerate(row_data):
                        cells[i].text = text

            # ------------------ MỤC 6 ------------------
            doc.add_heading("6. Biểu đồ kết quả với tệp dữ liệu mới", level=2)
            for img in ["prediction_bar_new.png", "confusion_matrix_new.png"]:
                path = os.path.join(output_dir, img)
                if os.path.exists(path):
                    doc.add_picture(path, width=Inches(5.5))
                    doc.add_paragraph("")

            # Create advanced filename
            now_str = datetime.now().strftime("%d-%m-%Y")
            counter_str = str(report_counter).zfill(5)
            report_filename = f"Report-{counter_str}-Hiep-{token_name}-XGB-{now_str}.docx"
            report_path = os.path.join(output_dir, report_filename)
            doc.save(report_path)
            print(f"📄 Đã tạo báo cáo Word: {report_path}")

        # Generate Word report at the end of each iteration
        generate_report_word(81)