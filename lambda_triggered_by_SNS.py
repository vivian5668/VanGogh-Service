import json
import boto3

from io import BytesIO
import smtplib
import imghdr
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def lambda_handler(event, context):
    information = json.loads(event['Records'][0]['Sns']['Message'])['Records'][0]
    s3_information = information['s3']
    bucket_name = s3_information['bucket']['name']
    object_name = s3_information['object']['key']
    uuid = object_name.split('/')[1].split('.')[0]
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('styleTransfer')
    item = table.get_item(Key={"requestId":uuid})['Item']
    email = item['email']
    user = item['name']
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, object_name)
    

