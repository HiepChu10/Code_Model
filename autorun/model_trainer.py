import time
import gc
from xgboost import XGBClassifier
from utils import clear_memory, check_memory_threshold


class ModelTrainer:
    def __init__(self, xgb_params, memory_settings=None):
        self.xgb_params = xgb_params
        self.memory_settings = memory_settings or {}
        
    def calculate_scale_pos_weight(self, y_train, log_print=None):
        """Calculate scale_pos_weight for imbalanced data"""
        try:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            if log_print:
                log_print(f"⚖️  Scale pos weight: {scale_pos_weight:.3f}")
            return float(scale_pos_weight)
        except Exception as e:
            if log_print:
                log_print(f"⚠️  Error calculating scale_pos_weight: {str(e)}, using default 1.0")
            return 1.0
    
    def train_model(self, X_train, y_train, log_print=None):
        """Train XGBoost model with memory optimization"""
        if log_print:
            log_print("🚀 Starting model training...")
        
        # Check memory before training
        if check_memory_threshold(threshold_percent=85, log_print=log_print):
            if log_print:
                log_print("🧹 Clearing memory before training...")
            clear_memory(log_print)
        
        try:
            # Calculate scale_pos_weight
            scale_pos_weight = self.calculate_scale_pos_weight(y_train, log_print)
        
            # Create model with scale_pos_weight and memory-optimized settings
            model_params = self.xgb_params.copy()
            model_params['scale_pos_weight'] = scale_pos_weight
        
            # Reduce n_jobs if memory is constrained
            if check_memory_threshold(threshold_percent=80):
                model_params['n_jobs'] = min(model_params.get('n_jobs', 4), 2)
                if log_print:
                    log_print(f"🔧 Reduced n_jobs to {model_params['n_jobs']} due to memory constraints")
            
            # Use tree_method='hist' for better memory efficiency with large datasets
            if len(X_train) > 100000:
                model_params['tree_method'] = 'hist'
                # Enable single precision for float32 optimization
                model_params['single_precision_histogram'] = True
                if log_print:
                    log_print("🌳 Using histogram-based tree method with float32 optimization")
            
            xgb_model = XGBClassifier(**model_params)
        
            if log_print:
                log_print(f"🎯 Training on {len(X_train):,} samples with {len(X_train.columns)} features")
            
            # Training with timing
            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            elapsed = time.time() - start_time
        
            if log_print:
                log_print(f"⏱️ Training completed in {elapsed:.2f} seconds")
            
            # Clear memory after training if enabled
            if self.memory_settings.get('CLEAR_MEMORY', True):
                gc.collect()
            
            return xgb_model, scale_pos_weight, elapsed
            
        except MemoryError as e:
            if log_print:
                log_print(f"❌ Memory error during training: {str(e)}")
                log_print("🔄 Trying with reduced memory settings...")
            
            # Retry with more aggressive memory settings
            return self._train_with_reduced_memory(X_train, y_train, log_print)
            
        except Exception as e:
            if log_print:
                log_print(f"❌ Error during training: {str(e)}")
            raise
    
    def _train_with_reduced_memory(self, X_train, y_train, log_print=None):
        """Train model with aggressive memory optimization"""
        if log_print:
            log_print("🔧 Applying aggressive memory optimization...")
        
        # Clear memory aggressively
        clear_memory(log_print)
        
        # Calculate scale_pos_weight
        scale_pos_weight = self.calculate_scale_pos_weight(y_train, log_print)
        
        # Use very conservative settings
        model_params = {
            'n_estimators': min(self.xgb_params.get('n_estimators', 300), 100),
            'learning_rate': self.xgb_params.get('learning_rate', 0.05),
            'max_depth': min(self.xgb_params.get('max_depth', 8), 6),
            'min_child_weight': max(self.xgb_params.get('min_child_weight', 2), 5),
            'subsample': self.xgb_params.get('subsample', 0.8),
            'colsample_bytree': self.xgb_params.get('colsample_bytree', 0.8),
            'reg_alpha': self.xgb_params.get('reg_alpha', 0.01),
            'reg_lambda': self.xgb_params.get('reg_lambda', 0.1),
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.xgb_params.get('random_state', 42),
            'eval_metric': self.xgb_params.get('eval_metric', 'logloss'),
            'verbosity': 0,
            'n_jobs': 1,  # Single thread to save memory
            'tree_method': 'hist',  # More memory efficient
            'max_bin': 64,  # Reduce memory usage
            'single_precision_histogram': True  # Float32 optimization
        }
        
        if log_print:
            log_print(f"🎯 Reduced training: n_estimators={model_params['n_estimators']}, "
                     f"max_depth={model_params['max_depth']}, n_jobs=1")
        
        try:
            xgb_model = XGBClassifier(**model_params)
            
            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            elapsed = time.time() - start_time
            
            if log_print:
                log_print(f"⏱️ Reduced training completed in {elapsed:.2f} seconds")
        
            return xgb_model, scale_pos_weight, elapsed 
            
        except Exception as e:
            if log_print:
                log_print(f"❌ Failed even with reduced settings: {str(e)}")
            raise 