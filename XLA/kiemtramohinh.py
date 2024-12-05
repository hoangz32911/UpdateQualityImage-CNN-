import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Hàm kiểm tra thông tin của mô hình đã huấn luyện
def check_trained_model(model_path):
    try:
        # Tải mô hình đã huấn luyện từ file
        model = load_model(model_path)
        print(f"Mô hình đã được tải thành công từ: {model_path}")
        
        # In ra cấu trúc của mô hình (toàn bộ mô hình và các lớp con bên trong nếu có)
        model.summary()
        
        # Kiểm tra xem mô hình có phải là một mô hình hợp nhất với nhiều phần (sub-model) không
        if isinstance(model, tf.keras.Model):
            print("\nCác lớp trong mô hình chính:")
            for layer in model.layers:
                print(f"- Lớp: {layer.name}, Kiểu: {type(layer)}")
        
        # Nếu mô hình có các mô hình con (sub-models), in ra thông tin của chúng
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                if isinstance(layer, tf.keras.Model):  # Kiểm tra nếu lớp là một sub-model
                    print(f"\nMô hình con tại lớp {i}: {layer.name}")
                    layer.summary()
                    
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

# Đường dẫn đến mô hình đã huấn luyện
model_path = "super_resolution_cnn.h5"  # Đảm bảo rằng đường dẫn tới mô hình là chính xác

# Kiểm tra mô hình đã huấn luyện
check_trained_model(model_path)
