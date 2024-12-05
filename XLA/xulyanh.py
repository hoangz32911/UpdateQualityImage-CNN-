import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Đọc ảnh và tạo ảnh độ phân giải thấp (Low Resolution)
def load_and_preprocess_image(image_path):
    # Đọc ảnh gốc (High Resolution)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tạo ảnh độ phân giải thấp (Low Resolution) bằng cách giảm kích thước ảnh gốc
    image_lr = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)

    return image, image_lr

# 2. Nâng cấp ảnh bằng mô hình đã huấn luyện
def upscale_image(model, image_lr):
    # Chuẩn hóa ảnh độ phân giải thấp (Low Resolution)
    x_input = np.array([image_lr]).astype('float32') / 255.0
    
    # Dự đoán ảnh độ phân giải cao từ mô hình
    predicted_image = model.predict(x_input)
    
    # Khôi phục giá trị pixel của ảnh dự đoán (từ [0, 1] về [0, 255])
    predicted_image = predicted_image * 255.0
    predicted_image = np.clip(predicted_image, 0, 255).astype('uint8')
    
    return predicted_image

# 3. Nâng cấp một ảnh và hiển thị kết quả
def upgrade_image(image_path, model_path):
    # Bước 1: Tải mô hình đã huấn luyện
    model = tf.keras.models.load_model(model_path)

    # Bước 2: Đọc và tiền xử lý ảnh
    image, image_lr = load_and_preprocess_image(image_path)

    # Bước 3: Nâng cấp ảnh (Super-Resolution)
    upgraded_image = upscale_image(model, image_lr)

    # Bước 4: Hiển thị kết quả
    plt.figure(figsize=(15, 5))

    # Ảnh gốc
    plt.subplot(1, 3, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image)
    plt.axis("off")

    # Ảnh độ phân giải thấp
    plt.subplot(1, 3, 2)
    plt.title("Ảnh độ phân giải thấp")
    plt.imshow(image_lr)
    plt.axis("off")

    # Ảnh sau khi nâng cấp
    plt.subplot(1, 3, 3)
    plt.title("Ảnh sau khi nâng cấp")
    plt.imshow(upgraded_image[0])
    plt.axis("off")

    plt.show()

    # Lưu ảnh sau khi nâng cấp (tùy chọn)
    output_path = "upgraded_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, cv2.cvtColor(upgraded_image[0], cv2.COLOR_RGB2BGR))
    print(f"Đã lưu ảnh nâng cấp tại: {output_path}")

# 4. Chạy hàm nâng cấp ảnh
if __name__ == "__main__":
    image_path = "anhthap.jpg"  # Đường dẫn đến ảnh cần nâng cấp
    model_path = "super_resolution_cnn.h5"  # Đường dẫn đến mô hình đã huấn luyện
    upgrade_image(image_path, model_path)
