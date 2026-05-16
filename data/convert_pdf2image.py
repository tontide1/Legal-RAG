import os
from pdf2image import convert_from_path

input_folder = "data"
output_folder = os.path.join(input_folder, "images")

# tạo thư mục images nếu chưa có
os.makedirs(output_folder, exist_ok=True)
poppler_path = r"poppler-24.11.0\Library\bin"
# duyệt tất cả file trong thư mục data
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file_name)

        # bỏ phần .pdf để lấy tên file
        base_name = os.path.splitext(file_name)[0]

        # convert PDF thành list ảnh
        images = convert_from_path(pdf_path, poppler_path=poppler_path)

        # lưu từng trang
        for i, img in enumerate(images):
            output_path = os.path.join(
                output_folder, f"{base_name}_page_{i+1}.png"
            )
            img.save(output_path, "PNG")

print("Done!")