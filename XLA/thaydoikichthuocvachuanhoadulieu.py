# Import thư viện
import tensorflow as tf
import matplotlib.pyplot as plt

# Đường dẫn tới ảnh
image_path = "anhthap.jpg"  # Đặt đường dẫn tới ảnh của bạn

# Tải ảnh
image = tf.io.read_file(image_path)  # Đọc file ảnh
image = tf.image.decode_jpeg(image, channels=3)  # Giải mã ảnh JPEG và giữ nguyên 3 kênh màu (RGB)

# Thay đổi kích thước ảnh (phóng to gấp 2 lần với nội suy bicubic)
upsampled_image = tf.image.resize(image, 
                                   [image.shape[0] * 2, image.shape[1] * 2],  # Gấp đôi chiều cao và chiều rộng
                                   method='bicubic')  # Sử dụng nội suy bicubic để nâng cao chất lượng

# Hiển thị ảnh gốc và ảnh đã phóng to
plt.figure(figsize=(10, 5))  # Thiết lập kích thước khung hình

# Ảnh gốc
plt.subplot(1, 2, 1)  # Tạo biểu đồ đầu tiên trong hình với 1 hàng, 2 cột
plt.title("Ảnh Gốc")  # Tiêu đề
plt.imshow(image.numpy())  # Hiển thị ảnh gốc
plt.axis("off")  # Ẩn các trục

# Ảnh nâng cao
plt.subplot(1, 2, 2)  # Tạo biểu đồ thứ hai
plt.title("Ảnh Nâng Cao (Bicubic)")  # Tiêu đề
plt.imshow(upsampled_image.numpy().astype("uint8"))  # Hiển thị ảnh đã phóng to
plt.axis("off")  # Ẩn các trục

plt.show()  # Hiển thị kết quả
