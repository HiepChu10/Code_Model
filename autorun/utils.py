import pandas as pd
import numpy as np
import os
import gc
import psutil


def format_int(n):
    """Format integer with thousands separator"""
    return f"{int(n):,}" if pd.notna(n) and n is not None else "-"


def format_percentage(n):
    """Format number as percentage"""
    return f"{n:.2%}" if pd.notna(n) and n is not None else "-"


def setup_logging(output_dir):
    """Setup logging to file"""
    log_file = open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8")
    
    def log_print(msg):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()  # Ensure immediate write
        
    return log_print, log_file


def get_token_name(file_path):
    """Extract token name from file path"""
    return os.path.basename(file_path).split("_")[0].upper()


def create_output_dir(label, feat_idx):
    """Create output directory for results"""
    output_dir = os.path.join("output", label, f"featureset_{feat_idx+1}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir 


def optimize_dtypes(df, log_print=None):
    """Optimize DataFrame dtypes to save memory - AGGRESSIVE FLOAT32 optimization"""
    if log_print:
        log_print("🔧 Optimizing data types to save memory (using float32)...")
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if df[col].dtype == np.int64:
            # Integer optimization
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
        elif df[col].dtype in [np.float64, np.float32]:
            # AGGRESSIVE: Always use float32 for floating point
            # float32 has ~7 decimal digits precision, sufficient for most ML features
            df[col] = df[col].astype(np.float32)
            
    new_memory = df.memory_usage(deep=True).sum() / 1024**2
    saved_mb = original_memory - new_memory
    saved_percent = (saved_mb / original_memory) * 100 if original_memory > 0 else 0
    
    if log_print:
        log_print(f"💾 Memory optimized: {original_memory:.1f}MB → {new_memory:.1f}MB")
        log_print(f"💰 Saved: {saved_mb:.1f}MB ({saved_percent:.1f}%) - Using float32")
    
    return df


def convert_to_float32(X_train, X_test, log_print=None):
    """Convert training data to float32 for XGBoost training - 50% memory savings"""
    if log_print:
        log_print("🔄 Converting training data to float32...")
    
    # Get memory before conversion
    train_memory_before = X_train.memory_usage(deep=True).sum() / 1024**2
    test_memory_before = X_test.memory_usage(deep=True).sum() / 1024**2
    
    # Convert all numeric columns to float32
    for col in X_train.select_dtypes(include=[np.number]).columns:
        X_train[col] = X_train[col].astype(np.float32)
        X_test[col] = X_test[col].astype(np.float32)
    
    # Get memory after conversion
    train_memory_after = X_train.memory_usage(deep=True).sum() / 1024**2
    test_memory_after = X_test.memory_usage(deep=True).sum() / 1024**2
    
    total_saved = (train_memory_before + test_memory_before) - (train_memory_after + test_memory_after)
    
    if log_print:
        log_print(f"📊 Training data memory: {train_memory_after:.1f}MB (train) + {test_memory_after:.1f}MB (test)")
        log_print(f"💰 Float32 conversion saved: {total_saved:.1f}MB")
    
    return X_train, X_test


def clear_memory(log_print=None):
    """Clear memory and run garbage collection"""
    gc.collect()
    if log_print:
        memory_info = psutil.virtual_memory()
        log_print(f"🧹 Memory cleared. Available: {memory_info.available / 1024**3:.1f}GB "
                 f"({memory_info.percent:.1f}% used)")


def get_memory_usage():
    """Get current memory usage"""
    memory_info = psutil.virtual_memory()
    return {
        'total_gb': memory_info.total / 1024**3,
        'available_gb': memory_info.available / 1024**3,
        'used_percent': memory_info.percent
    }


def check_memory_threshold(threshold_percent=85, log_print=None):
    """Check if memory usage exceeds threshold"""
    memory_info = get_memory_usage()
    if memory_info['used_percent'] > threshold_percent:
        if log_print:
            log_print(f"⚠️  High memory usage: {memory_info['used_percent']:.1f}% "
                     f"(Available: {memory_info['available_gb']:.1f}GB)")
        return True
    return False



def estimate_memory_savings(total_records, num_features, log_print=None):
    """Estimate memory savings from float32 optimization"""
    # Estimate memory usage
    float64_size = total_records * num_features * 8 / 1024**2  # MB
    float32_size = total_records * num_features * 4 / 1024**2  # MB
    savings = float64_size - float32_size
    
    if log_print:
        log_print(f"📊 Estimated memory usage:")
        log_print(f"   Float64: {float64_size:.1f}MB")
        log_print(f"   Float32: {float32_size:.1f}MB")
        log_print(f"   Savings: {savings:.1f}MB ({savings/float64_size*100:.1f}%)")
    
    return float32_size, savings 