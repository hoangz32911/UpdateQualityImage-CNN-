import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 1. Đọc ảnh và tạo ảnh độ phân giải thấp (Low Resolution)
def load_and_preprocess_image(image_path):
    # Đọc ảnh gốc (High Resolution)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tạo ảnh độ phân giải thấp (Low Resolution) bằng cách giảm kích thước ảnh gốc
    image_lr = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)

    return image, image_lr

# 2. Xây dựng mô hình CNN để nâng cấp độ phân giải ảnh (Super Resolution)
def build_cnn_super_resolution():
    model = models.Sequential([
        layers.InputLayer(input_shape=(None, None, 3)),  # Đầu vào là ảnh màu RGB
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),  # Upsample ảnh về kích thước gốc

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),  # Upsample ảnh về kích thước gốc

        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Đầu ra là ảnh RGB
    ])
    return model

# 3. Huấn luyện mô hình CNN để nâng cấp độ phân giải
def train_model(model, x_train, y_train, epochs=10, batch_size=1):
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    print("Huấn luyện hoàn thành!")

# 4. Đánh giá và hiển thị kết quả
def evaluate_and_display_results(model, x_test, y_test):
    predicted_image = model.predict(x_test)
    
    # Khôi phục giá trị pixel của ảnh dự đoán (từ [0, 1] về [0, 255])
    predicted_image = predicted_image * 255.0  # Khôi phục lại giá trị pixel
    predicted_image = np.clip(predicted_image, 0, 255).astype('uint8')  # Đảm bảo giá trị trong phạm vi [0, 255]

    # Resize ảnh dự đoán về kích thước ảnh gốc
    predicted_image_resized = cv2.resize(predicted_image[0], (y_test.shape[2], y_test.shape[1]), interpolation=cv2.INTER_CUBIC)

    plt.figure(figsize=(15, 5))

    # Ảnh gốc
    plt.subplot(1, 3, 1)
    plt.title("Ảnh độ phân giải cao")
    plt.imshow(y_test[0])
    plt.axis("off")

    # Ảnh độ phân giải thấp
    plt.subplot(1, 3, 2)
    plt.title("Ảnh độ phân giải thấp")
    plt.imshow(x_test[0])
    plt.axis("off")

    # Ảnh sau khi nâng cấp (super-resolved)
    plt.subplot(1, 3, 3)
    plt.title("Ảnh sau khi nâng cấp")
    plt.imshow(predicted_image_resized)
    plt.axis("off")

    plt.show()

# 5. Hàm chính để thực hiện huấn luyện
def main(image_path):
    # Bước 1: Tiền xử lý dữ liệu
    image, image_lr = load_and_preprocess_image(image_path)

    # Resize both low-resolution and high-resolution images to the same fixed size
    target_size = (256, 256)  # Resize both images to this fixed size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    image_lr_resized = cv2.resize(image_lr, target_size, interpolation=cv2.INTER_CUBIC)

    # Chuyển ảnh thành định dạng numpy array và chuẩn hóa
    x_train = np.array([image_lr_resized]).astype('float32') / 255.0
    y_train = np.array([image]).astype('float32') / 255.0

    # Bước 2: Xây dựng mô hình CNN để nâng cấp ảnh
    model = build_cnn_super_resolution()

    # Bước 3: Huấn luyện mô hình
    train_model(model, x_train, y_train)

    # Bước 4: Lưu mô hình sau khi huấn luyện
    model.save('super_resolution_cnn.h5')
    print("Mô hình đã được lưu thành công!")

    # Bước 5: Đánh giá mô hình và hiển thị kết quả
    evaluate_and_display_results(model, x_train, y_train)

if __name__ == "__main__":
    # Đặt đường dẫn tới ảnh của bạn ở đây
    image_path = "anhthap.jpg"  # Thay thế bằng đường dẫn đúng đến ảnh của bạn
    main(image_path)
