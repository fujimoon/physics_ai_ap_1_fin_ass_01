# ROS 2 Humble + Ignition Gazebo Fortress (参考記事と同じ組み合わせ) を Apple Silicon / x86 どちらでも動かすための環境
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-gz \
      xvfb mesa-utils libgl1-mesa-dri \
      python3-pip python3-numpy python3-opencv python3-matplotlib \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ソフトウェアレンダリング（GPU無しのコンテナで Gazebo のカメラ/LiDAR を動かす）
ENV LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1 \
    IGN_IP=127.0.0.1

WORKDIR /work
CMD ["bash"]
