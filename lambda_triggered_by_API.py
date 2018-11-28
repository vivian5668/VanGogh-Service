import json
import boto3 
import uuid
import datetime
import base64

def lambda_handler(event, context):
    # TODO implement
    new_event = json.loads(event['body'])
    print(new_event)
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('styleTransfer')
    uuId = str(uuid.uuid4())
    while "Item" in table.get_item(Key={'requestId': uuId}):
        uuId = str(uuid.uuid4())
    s3 = boto3.resource('s3')
    object = s3.Object('chelsea-style-transfer', 'content/{}.jpg'.format(uuId))
    content_image_encoding = new_event['content_image_encoding']
    object.put(Body=base64.b64decode(str.encode(content_image_encoding[23:])))
    item = {
        'requestId': uuId,
            'timeStamp': str(datetime.datetime.utcnow()),

                'contentImageEncoding': "s3://chelsea-style-transfer/content/{}.jpg".format(uuId),
                'style_image': "s3://chelsea-style-transfer/{}.jpg".format(new_event['style_image']),  
                'email': new_event['email'],
                'name': new_event['name']
        
    }
    print(item)
    table.put_item(
        Item=item
    )

    style_image_path = new_event['style_image']
    output_image_path = "s3://chelsea-style-transfer/output/{}.jpg".format(uuId)
    client = boto3.client('ecs', region_name='us-west-2')
    response = client.run_task(
        cluster='style-transfer',
        launchType='FARGATE',
        taskDefinition='style-transfer',
        count = 1,
        platformVersion='LATEST',
        overrides={
        'containerOverrides': [
            {
                'name': "style-transfer",
                'environment': [
                    {
                        'name': 'AWS_DEFAULT_REGION',
                        'value': 'us-west-2'
                    },
                    {
                        'name': 'CONTENT_IMAGE_PATH',
                        'value': "s3://chelsea-style-transfer/content/{}.jpg".format(uuId)
                    },
                    {
                        'name': 'STYLE_IMAGE_PATH',
                        'value': 's3://chelsea-style-transfer/{}.jpg'.format(style_image_path)
                    },
                    {
                        'name': 'OUTPUT_IMAGE_PATH',
                        'value': output_image_path
                    }
                ]
            }
        ]
           },
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets':[
                    'subnet-f1b00e88'
                    ],
                'assignPublicIp': 'ENABLED'
                }}
        )
    

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully Submitted!')
    }
