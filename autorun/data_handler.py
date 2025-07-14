import pandas as pd
from sklearn.model_selection import train_test_split
from utils import optimize_dtypes, clear_memory, check_memory_threshold


class DataHandler:
    """
    Handles data loading, preprocessing, and management with memory optimization.
    
    Features:
    - Memory-optimized data loading
    - Automatic chunked reading for large datasets
    - Data type optimization (float32)
    - Sampling for memory-constrained environments
    - Memory monitoring and cleanup
    """
    
    def __init__(self, data_path, test_new_path, memory_settings=None):
        """
        Initialize DataHandler.
        
        Args:
            data_path (str): Path to main training data CSV
            test_new_path (str): Path to new test data CSV
            memory_settings (dict): Memory optimization settings
        """
        self.data_path = data_path
        self.test_new_path = test_new_path
        self.memory_settings = memory_settings or {}
    
    def load_main_data(self, y_type, features, log_print=None):
        """
        Load main training data with memory optimization.
        
        Args:
            y_type (str): Target column name
            features (list): List of feature column names
            log_print (function): Logging function
            
        Returns:
            tuple: (X, y, count_0, count_1, total_count)
        """
        if log_print:
            log_print("📁 Loading main data...")
        
        try:
            # First, try to read normally
            df = pd.read_csv(self.data_path)
            
            if log_print:
                log_print(f"📊 Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Optimize dtypes if enabled
            if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                df = optimize_dtypes(df, log_print)
            
            # Check memory usage
            check_memory_threshold(log_print=log_print)
            
            # Convert OpenTime and set index
            df['OpenTime'] = pd.to_datetime(df['OpenTime'])
            df.set_index('OpenTime', inplace=True)
            
            # Data statistics
            count_0 = (df[y_type] == 0).sum()
            count_1 = (df[y_type] == 1).sum()
            total_count = len(df)
            
            # Extract features and target
            X = df[features].copy()
            y = df[y_type].copy()
            
            # Optimize X to float32 immediately after extraction
            if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                for col in X.select_dtypes(include=['float64']).columns:
                    X[col] = X[col].astype('float32')
                if log_print:
                    log_print("🔧 Features converted to float32")
            
            # Clear intermediate variables
            del df
            if self.memory_settings.get('CLEAR_MEMORY', True):
                clear_memory(log_print)
            
            return X, y, count_0, count_1, total_count
            
        except MemoryError as e:
            if log_print:
                log_print(f"❌ Memory error: {str(e)}")
                log_print("🔄 Trying chunked reading...")
            
            return self._load_data_chunked(y_type, features, log_print)
    
    def _load_data_chunked(self, y_type, features, log_print=None):
        """
        Load data in chunks to handle memory constraints.
        
        Args:
            y_type (str): Target column name
            features (list): List of feature column names
            log_print (function): Logging function
            
        Returns:
            tuple: (X, y, count_0, count_1, total_count)
        """
        chunk_size = self.memory_settings.get('CHUNK_SIZE', 100000)
        
        if log_print:
            log_print(f"📊 Loading data in chunks of {chunk_size:,} rows...")
        
        chunks = []
        total_rows = 0
        
        try:
            # Read in chunks
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                    chunk = optimize_dtypes(chunk)
                
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if log_print and len(chunks) % 10 == 0:
                    log_print(f"📊 Loaded {total_rows:,} rows...")
            
            # Combine chunks
            if log_print:
                log_print("🔗 Combining chunks...")
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            clear_memory(log_print)
            
            # Convert OpenTime and set index
            df['OpenTime'] = pd.to_datetime(df['OpenTime'])
            df.set_index('OpenTime', inplace=True)
            
            # Data statistics
            count_0 = (df[y_type] == 0).sum()
            count_1 = (df[y_type] == 1).sum()
            total_count = len(df)
            
            # Extract features and target
            X = df[features].copy()
            y = df[y_type].copy()
            
            # Optimize X to float32
            if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                for col in X.select_dtypes(include=['float64']).columns:
                    X[col] = X[col].astype('float32')
                if log_print:
                    log_print("🔧 Features converted to float32 (chunked)")
            
            # Clear intermediate variables
            del df
            clear_memory(log_print)
            
            return X, y, count_0, count_1, total_count
            
        except Exception as e:
            if log_print:
                log_print(f"❌ Error in chunked loading: {str(e)}")
            raise
    
    def split_data(self, X, y, test_size=0.3, random_state=42, log_print=None):
        """
        Split data into train and test sets.
        
        Args:
            X (DataFrame): Features
            y (Series): Target variable
            test_size (float): Proportion of test set (default: 0.3)
            random_state (int): Random seed for reproducibility
            log_print (function): Logging function
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if log_print:
            log_print("✂️  Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state
        )
        
        if log_print:
            log_print(f"📊 Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def load_new_test_data(self, y_type, features, log_print=None):
        """
        Load new test data with memory optimization.
        
        Args:
            y_type (str): Target column name
            features (list): List of feature column names
            log_print (function): Logging function
            
        Returns:
            tuple: (X_new, y_new, new_data_info)
        """
        if log_print:
            log_print("📁 Loading new test data...")
        
        try:
            # Load new test data
            df_new = pd.read_csv(self.test_new_path)
            
            # Optimize dtypes if enabled
            if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                df_new = optimize_dtypes(df_new, log_print)
            
            # Calculate statistics
            count_rows_new = len(df_new)
            count_0 = (df_new[y_type] == 0).sum()
            count_1 = (df_new[y_type] == 1).sum()
            pct_0 = count_0 / count_rows_new * 100 if count_rows_new > 0 else 0
            pct_1 = count_1 / count_rows_new * 100 if count_rows_new > 0 else 0
            
            new_data_info = {
                'count_rows_new': count_rows_new,
                'count_0': count_0,
                'count_1': count_1,
                'pct_0': pct_0,
                'pct_1': pct_1
            }
            
            # Check if target column exists
            if y_type in df_new.columns:
                y_new = df_new[y_type].copy()
                X_new = df_new[features].copy()
                
                # Optimize to float32
                if self.memory_settings.get('OPTIMIZE_DTYPES', True):
                    for col in X_new.select_dtypes(include=['float64']).columns:
                        X_new[col] = X_new[col].astype('float32')
                    if log_print:
                        log_print("🔧 New test data converted to float32")
                
                # Clear intermediate variables
                del df_new
                if self.memory_settings.get('CLEAR_MEMORY', True):
                    clear_memory(log_print)
                
                return X_new, y_new, new_data_info
            else:
                # Target column doesn't exist
                if log_print:
                    log_print(f"⚠️  Target column '{y_type}' not found in new test data")
                
                del df_new
                if self.memory_settings.get('CLEAR_MEMORY', True):
                    clear_memory(log_print)
                
                return None, None, new_data_info
                
        except Exception as e:
            if log_print:
                log_print(f"❌ Error loading new test data: {str(e)}")
            
            # Return empty info on error
            return None, None, {
                'count_rows_new': 0, 
                'count_0': 0, 
                'count_1': 0, 
                'pct_0': 0, 
                'pct_1': 0
            }
    
    def get_train_distribution(self, y_train, log_print):
        """
        Get and log training data distribution.
        
        Args:
            y_train (Series): Training target variable
            log_print (function): Logging function
            
        Returns:
            Series: Distribution percentages
        """
        if log_print:
            log_print("\n📊 Tỷ lệ nhãn tập train:")
            
        train_dist = y_train.value_counts(normalize=True) * 100
        
        if log_print:
            for label, pct in train_dist.items():
                log_print(f"   Nhãn {label}: {pct:.1f}%")
        
        return train_dist 