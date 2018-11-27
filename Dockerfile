from amazonlinux
#
# written for Amazon Linux AMI
# creates an AWS Lambda deployment package for pytorch deep learning models (Python 3.6.1)
# assumes lambda function defined in ~/main.py
# deployment package created at ~/waya-ai-lambda.zip
#


RUN yum install -y python3
RUN rm -rf /var/cache/yum


RUN pip3 --no-cache-dir install Pillow numpy torch torchvision boto3
COPY vgg.py /bin/vgg.py
RUN python3 /bin/vgg.py

COPY main.py /bin/main.py

ENTRYPOINT ["/bin/python3", "/bin/main.py"]


