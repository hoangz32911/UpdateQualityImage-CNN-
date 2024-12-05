import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Kiểm tra mô hình
def check_model(model_path):
    try:
        model = load_model(model_path)  # Tải mô hình đã huấn luyện
        print("Mô hình đã được tải thành công.")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

# Kiểm tra mô hình với một ảnh mẫu
def test_model_on_sample_image(model, image_path):
    # Đọc ảnh và tiền xử lý
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Kiểm tra nếu ảnh có kích thước hợp lệ (chia hết cho 4)
    if image.shape[0] % 4 != 0 or image.shape[1] % 4 != 0:
        print("Ảnh không hợp lệ, kích thước không chia hết cho 4.")
        return
    
    # Tiền xử lý ảnh cho mô hình
    image_lr_resized = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
    x_test = np.array([image_lr_resized]).astype('float32') / 255.0
    
    # Dự đoán ảnh siêu phân giải
    predicted_image = model.predict(x_test)
    
    # Kiểm tra nếu ảnh dự đoán có kích thước hợp lệ
    if predicted_image.shape[1] != image.shape[0] or predicted_image.shape[2] != image.shape[1]:
        print("Lỗi: Kích thước ảnh dự đoán không khớp với ảnh gốc.")
        return
    
    # Chuyển giá trị pixel từ [0, 1] về [0, 255]
    predicted_image = predicted_image * 255.0
    predicted_image = np.clip(predicted_image, 0, 255).astype('uint8')
    
    print("Mô hình dự đoán thành công.")
    
    return predicted_image[0]

# Đường dẫn đến mô hình và ảnh mẫu
model_path = "super_resolution_cnn.h5"
image_path = "upgraded_anhthap.jpg"  # Thay bằng ảnh mẫu có độ phân giải thấp

# Kiểm tra mô hình
model = check_model(model_path)

# Nếu mô hình tải thành công, kiểm tra ảnh mẫu
if model:
    predicted_image = test_model_on_sample_image(model, image_path)
    if predicted_image is not None:
        # Hiển thị ảnh gốc và ảnh siêu phân giải
        import matplotlib.pyplot as plt
        
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 5))
        
        # Ảnh gốc
        plt.subplot(1, 2, 1)
        plt.title("Ảnh Gốc")
        plt.imshow(original_image)
        plt.axis("off")
        
        # Ảnh dự đoán
        plt.subplot(1, 2, 2)
        plt.title("Ảnh Siêu Phân Giải")
        plt.imshow(predicted_image)
        plt.axis("off")
        
        plt.show()
