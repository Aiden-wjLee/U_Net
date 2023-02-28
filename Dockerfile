#https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html#rel_22-04
FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN pip install torchvision
RUN pip install numpy
RUN pip install Pillow
RUN pip install matplotlib
#
RUN pip install pycocotools
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install pandas

RUN export TORCH_CUDA_ARCH_LIST="compute capability"
RUN export CUDA_HOME=/usr/local/cuda-11.2


WORKDIR /home/

COPY . .

CMD ["python","mIoU_show.py"]