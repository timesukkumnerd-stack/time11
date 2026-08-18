import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. โหลดภาพสีและทำ DFT (แปลงเป็นโดเมนความถี่)
# ==========================================

# เปลี่ยนชื่อไฟล์ให้ตรงกับรูปน้องหมา
# สมมติว่าตั้งชื่อไฟล์ว่า dog.png
img = cv2.imread('time.jpg', 1)

if img is None:
    raise FileNotFoundError(
        "ไม่พบไฟล์ภาพ 'dog.png' กรุณาตรวจสอบชื่อไฟล์และที่อยู่ของไฟล์อีกครั้ง"
    )

# แปลงภาพเป็น Grayscale ชั่วคราวเฉพาะตอนทำ DFT
img_gray_for_dft = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_f = np.float32(img_gray_for_dft)

dft = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

# หา Magnitude Spectrum
magnitude = 20 * np.log(
    cv2.magnitude(
        dft_shift[:, :, 0],
        dft_shift[:, :, 1]
    ) + 1
)


# ==========================================
# 2. ทำ Mask และซ่อมภาพด้วย cv2.inpaint
# ==========================================

gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# จับวัตถุสีเข้ม / รอยดำ
_, my_mask = cv2.threshold(
    gray_temp,
    30,
    255,
    cv2.THRESH_BINARY_INV
)

# ซ่อมภาพด้วย Inpaint
img_repair = cv2.inpaint(
    img,
    my_mask,
    3,
    cv2.INPAINT_TELEA
)


# ==========================================
# 3. บีบอัดภาพ JPEG และคำนวณค่า PSNR
# ==========================================

# คุณภาพ JPEG 25%
quality = 25

params = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    quality
]

# บีบอัดภาพ
_, encoded = cv2.imencode(
    '.jpg',
    img_repair,
    params
)

# ถอดรหัสกลับมาเป็นภาพสี
img_compressed = cv2.imdecode(
    encoded,
    1
)

# คำนวณ MSE
mse = np.mean(
    (img_repair.astype(np.float32) -
     img_compressed.astype(np.float32)) ** 2
)

# คำนวณ PSNR
if mse == 0:
    psnr_result = 99.99
else:
    max_val = 255.0
    psnr_result = 20 * np.log10(
        max_val / np.sqrt(mse)
    )

print(f"ค่า PSNR ที่คำนวณได้ = {psnr_result:.2f} dB")


# ==========================================
# 4. พล็อตสรุปผลส่งแล็บ
# ==========================================

plt.figure(figsize=(12, 8))


# ------------------------------
# ภาพต้นฉบับ
# ------------------------------
plt.subplot(2, 3, 1)

plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
)

plt.title('Original Image (Perplexed Dog)')
plt.axis('off')


# ------------------------------
# DFT Spectrum
# ------------------------------
plt.subplot(2, 3, 2)

plt.imshow(
    magnitude,
    cmap='gray'
)

plt.title('DFT Spectrum')
plt.axis('off')


# ------------------------------
# Mask
# ------------------------------
plt.subplot(2, 3, 3)

plt.imshow(
    my_mask,
    cmap='gray'
)

plt.title('Adjusted Mask')
plt.axis('off')


# ------------------------------
# ภาพหลังซ่อม
# ------------------------------
plt.subplot(2, 3, 4)

plt.imshow(
    cv2.cvtColor(
        img_repair,
        cv2.COLOR_BGR2RGB
    )
)

plt.title('Inpainted Image')
plt.axis('off')


# ------------------------------
# ภาพหลังบีบอัด
# ------------------------------
plt.subplot(2, 3, 5)

plt.imshow(
    cv2.cvtColor(
        img_compressed,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(f'Compressed {quality}%')
plt.axis('off')


# ------------------------------
# แสดงผล PSNR
# ------------------------------
plt.subplot(2, 3, 6)

plt.text(
    0.1,
    0.5,
    f"Lab Results:\n\nPSNR = {psnr_result:.2f} dB",
    fontsize=14,
    color='darkblue',
    weight='bold'
)

plt.axis('off')


# จัดระยะห่าง
plt.tight_layout()
