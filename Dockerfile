#https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html#rel_22-04
FROM nvcr.io/nvidia/pytorch:22.04-py3



RUN pip install torchvision
RUN pip install numpy
RUN pip install Pillow
RUN pip install matplotlib
#
RUN pip install pycocotools

RUN apt-get update && apt-get -y install libgl1-mesa-glx
##RUN apt-get update && apt install libxcb-xinerama0
#RUN apt-get update && apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev 

#RUN pip install PyGLM PySide2 pyopengl
#RUN apt-get update && apt-get install -y libxcb-xinerama0
#RUN apt-get update && apt-get install -y libxcb-xinerama0-dev
ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get install -y --no-install-recommends libxcb-xinerama0-dev libxkbcommon-x11-0
RUN apt-get install -y libxcb-xinerama0-dev libxkbcommon-x11-0

#RUN apt update && apt -y install make g++ pkg-config libgl1-mesa-dev \
#    libxcb*-dev libfontconfig1-dev libxkbcommon-x11-dev python libgtk-3-dev
RUN apt install -y libgtk2.0-dev 
RUN apt install -y pkg-config 
#RUN pip install pyqt5
#RUN pip install pyqt5-tools

##RUN pip3 install opencv-contrib-python==4.5.5.62
RUN pip install opencv-python==4.1.2.30
#RUN pip3 install opencv-python-headless
RUN pip install scikit-image
RUN pip install pandas


ENV TORCH_CUDA_ARCH_LIST="compute capability"

#ENV CUDA_HOME=/usr/local/cuda-11.2
ENV DEBIAN_FRONTEND=newt
ENV DISPLAY=192.168.50.163:0
ENV QT_DEBUG_PLUGINS=1
ENV QT_X11_NO_MITSHM=1

WORKDIR /home/

#COPY . .

CMD ["python","mIoU_show.py"]