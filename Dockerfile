# Sử dụng base image Python 3.9 slim
FROM python:3.9-slim

# Cài đặt các công cụ và thư viện phụ thuộc cần thiết để biên dịch TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Tải và cài đặt TA-Lib C (phiên bản 0.4.0 từ SourceForge)
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O ta-lib-0.4.0.tar.gz && \
    tar -xzf ta-lib-0.4.0.tar.gz && \
    cd ta-lib
RUN set -x && \
    cd /tmp/ta-lib && \
    ./configure --prefix=/usr && \
    make clean && \
    make && \
    make install
# Kiểm tra và hiển thị kết quả cài đặt
RUN find / -name "libta-lib.so" || echo "Error: libta-lib.so not found" && \
    rm -rf /tmp/ta-lib*

# Cập nhật LD_LIBRARY_PATH để tìm thấy thư viện TA-Lib
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện Python cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt Python wrapper cho TA-Lib (phiên bản tương thích với TA-Lib 0.4.0)
RUN pip install TA-Lib==0.4.32

# Sao chép các file cần thiết vào container
COPY main.py .
COPY templates/ templates/

# Chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]