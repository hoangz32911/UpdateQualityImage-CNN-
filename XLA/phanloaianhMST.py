# Import thư viện
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Chuẩn bị dữ liệu
# Tải bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tiền xử lý dữ liệu
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Chuẩn hóa ảnh 0-1
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0    # Chuẩn hóa ảnh 0-1

# Chuyển đổi nhãn thành one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Xây dựng mô hình CNN
model = models.Sequential([
    # Lớp convolutional và max pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten và fully connected layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 lớp đầu ra
])

# Hiển thị kiến trúc mô hình
model.summary()

# 3. Biên dịch mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Huấn luyện mô hình
history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test))

# 5. Đánh giá mô hình
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Độ chính xác trên tập kiểm tra: {test_acc:.2f}")

# 6. Dự đoán trên một ảnh mẫu
sample_image = x_test[0]
prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
predicted_label = tf.argmax(prediction, axis=1).numpy()[0]

print(f"Nhãn dự đoán: {predicted_label}")
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {predicted_label}")
plt.axis("off")
plt.show()
