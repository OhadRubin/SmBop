FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
RUN apt-get update
RUN apt install -y libgraphviz-dev
ADD ./requirements.txt /workspace/requirements.txt
ADD ./demo/demo_req.txt /workspace/demo_req.txt
RUN pip install -r requirements.txt
RUN pip install -r demo_req.txt
COPY . /workspace/SmBop
WORKDIR /workspace/SmBop
RUN apt install -y unzip
RUN python demo/init.py
RUN unzip -q spider.zip 
RUN mv spider dataset
CMD ["python","-i","start_demo.py"]
