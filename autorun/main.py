#!/usr/bin/env python3
"""
Modular XGBoost Training Script with Memory Optimization
Processes cryptocurrency data to predict price movements using XGBoost.
"""

import os
import sys
import gc
import traceback
from config import (
    DATA_PATH, TEST_NEW_PATH, settings, feature_sets, 
    XGB_PARAMS, MEMORY_SETTINGS
)
from utils import (
    setup_logging, create_output_dir, clear_memory, 
    check_memory_threshold, get_memory_usage, convert_to_float32,
    estimate_memory_savings
)
from data_handler import DataHandler
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from report_generator import ReportGenerator


def main():
    """Main execution function with memory management"""
    print("=" * 60)
    print("🚀 XGBoost Training Pipeline with Memory Optimization")
    print("=" * 60)
    
    # Display memory info at start
    memory_info = get_memory_usage()
    print(f"💾 System Memory: {memory_info['total_gb']:.1f}GB total, "
          f"{memory_info['available_gb']:.1f}GB available ({memory_info['used_percent']:.1f}% used)")
    
    # Global results storage
    global_results = {}
    
    # Initialize components
    data_handler = DataHandler(DATA_PATH, TEST_NEW_PATH, MEMORY_SETTINGS)
    model_trainer = ModelTrainer(XGB_PARAMS, MEMORY_SETTINGS)
    report_generator = ReportGenerator(DATA_PATH, TEST_NEW_PATH)
    
    # Counter for successful/failed runs
    successful_runs = 0
    failed_runs = 0
    
    for feat_idx, features in enumerate(feature_sets):
        print(f"\n{'='*20} 🎯 FeatureSet {feat_idx+1} ({len(features)} features) {'='*20}")
        
        for config in settings:
            Y_TYPE = config['Y_TYPE']
            threshold = config['threshold']
            label = f"{Y_TYPE}_thr{threshold}"
            result_key = f"{label}_fs{feat_idx+1}"
            
            print(f"\n🔁 Running: {label} | FeatureSet {feat_idx+1}")
            
            # Create output directory
            output_dir = create_output_dir(label, feat_idx)
            
            # Setup logging
            log_print, log_file = setup_logging(output_dir)
            
            try:
                # Check memory before starting
                if check_memory_threshold(threshold_percent=80, log_print=log_print):
                    log_print("🧹 Clearing memory before processing...")
                    clear_memory(log_print)
                
                # ===== 1. DATA LOADING =====
                log_print(f"📊 Processing: {Y_TYPE} with threshold {threshold}")
                
                X, y, count_0, count_1, total_count = data_handler.load_main_data(
                    Y_TYPE, features, log_print
                )
                
                # Create full dataset stats for report
                full_dataset_stats = {
                    'count_0': count_0,
                    'count_1': count_1,
                    'total_count': total_count
                }
                
                log_print(f"Số lượng giá trị 0: {count_0:,}")
                log_print(f"Số lượng giá trị 1: {count_1:,}")
                log_print(f"Tổng số bản ghi: {total_count:,}")
                
                # ===== 2. DATA SPLITTING =====
                X_train, X_test, y_train, y_test = data_handler.split_data(
                    X, y, log_print=log_print
                )
                
                # Log train distribution
                data_handler.get_train_distribution(y_train, log_print)
                
                # ===== 2.5. CONVERT TO FLOAT32 FOR MEMORY OPTIMIZATION =====
                estimate_memory_savings(len(X_train), len(features), log_print)
                X_train, X_test = convert_to_float32(X_train, X_test, log_print)
                
                # Clear intermediate variables
                del X, y
                clear_memory(log_print)
                
                # ===== 3. MODEL TRAINING =====
                xgb_model, scale_pos_weight, elapsed = model_trainer.train_model(
                    X_train, y_train, log_print
                )
                
                # ===== 4. MODEL EVALUATION =====
                evaluator = ModelEvaluator(output_dir, MEMORY_SETTINGS)
                
                # Evaluate on test set
                test_report, test_acc = evaluator.evaluate_and_plot(
                    xgb_model, X_test, y_test, threshold, 'test', 'Tập Test', log_print
                )
                
                # Feature importance plot
                evaluator.plot_feature_importance(xgb_model, features, log_print)
                
                # ===== 5. NEW DATA EVALUATION =====
                X_new, y_new, new_data_info = data_handler.load_new_test_data(
                    Y_TYPE, features, log_print
                )
                
                log_print(f"\n📦 Dữ liệu mới: {new_data_info['count_rows_new']:,} bản ghi "
                         f"(0: {new_data_info['count_0']:,} - {new_data_info['pct_0']:.2f}%, "
                         f"1: {new_data_info['count_1']:,} - {new_data_info['pct_1']:.2f}%)")
                
                new_report = None
                new_acc = None
                if X_new is not None and y_new is not None:
                    new_report, new_acc = evaluator.evaluate_and_plot(
                        xgb_model, X_new, y_new, threshold, 'new', 'Tập Mới', log_print
                    )
                    
                    # Save metrics to CSV for tracking
                    if new_report is not None:
                        evaluator.save_metrics_to_csv(
                            new_report, Y_TYPE, threshold, feat_idx + 1, log_print
                        )
                else:
                    log_print("Không tìm thấy nhãn trong dữ liệu mới.")
                
                # ===== 6. STORE RESULTS =====
                global_results[result_key] = {
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
                    'output_dir': output_dir,
                    'report_test': test_report,
                    'accuracy_test': test_acc,
                    'new_data': new_data_info,
                    'full_dataset_stats': full_dataset_stats  # Add full dataset stats
                }
                
                if new_report is not None:
                    global_results[result_key]['report_new'] = new_report
                    global_results[result_key]['accuracy_new'] = new_acc
                
                # ===== 7. GENERATE WORD REPORT =====
                log_print("📄 Generating Word report...")
                report_path = report_generator.generate_word_report(
                    global_results[result_key], output_dir 
                )
                
                
                log_print(f"\n✅ Completed successfully: {result_key}")
                log_print(f"📁 Results saved to: {output_dir}")
                successful_runs += 1
                
                # Clear memory after each run
                if MEMORY_SETTINGS.get('CLEAR_MEMORY', True):
                    # Keep only essential data, clear large objects
                    del X_train, X_test, y_train, y_test, xgb_model
                    if X_new is not None:
                        del X_new, y_new
                    clear_memory(log_print)
                
            except MemoryError as e:
                error_msg = f"❌ Memory error in {result_key}: {str(e)}"
                print(error_msg)
                log_print(error_msg)
                log_print("💡 Try reducing chunk size or closing other applications")
                failed_runs += 1
                
                # Aggressive memory cleanup
                gc.collect()
                
            except Exception as e:
                error_msg = f"❌ Error processing {result_key}: {str(e)}"
                print(error_msg)
                log_print(error_msg)
                log_print(f"Traceback: {traceback.format_exc()}")
                failed_runs += 1
                
            finally:
                # Always close log file
                log_file.close()
    
                # Check memory status
                memory_info = get_memory_usage()
                print(f"💾 Memory after run: {memory_info['available_gb']:.1f}GB available "
                      f"({memory_info['used_percent']:.1f}% used)")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("📋 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total processed: {successful_runs + failed_runs}")
    
    if failed_runs > 0:
        print("\n💡 If you encountered memory errors, try:")
        print("   1. Reduce CHUNK_SIZE in config.py")
        print("   2. Increase system RAM")
        print("   3. Close other applications")
    
    memory_info = get_memory_usage()
    print(f"\n💾 Final memory: {memory_info['available_gb']:.1f}GB available "
          f"({memory_info['used_percent']:.1f}% used)")
    
    print("🎉 Pipeline completed!")
    return successful_runs, failed_runs


if __name__ == "__main__":
    try:
        successful, failed = main()
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 