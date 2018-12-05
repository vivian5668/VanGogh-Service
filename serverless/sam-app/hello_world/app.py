
import json
import boto3 
import uuid
import datetime
import base64

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        {
            "resource": "Resource path",
            "path": "Path parameter",
            "httpMethod": "Incoming request's method name"
            "headers": {Incoming request headers}
            "queryStringParameters": {query string parameters }
            "pathParameters":  {path parameters}
            "stageVariables": {Applicable stage variables}
            "requestContext": {Request context, including authorizer-returned key-value pairs}
            "body": "A JSON string of the request payload."
            "isBase64Encoded": "A boolean flag to indicate if the applicable request payload is Base64-encode"
        }

        https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

    Attributes
    ----------

    context.aws_request_id: str
         Lambda request ID
    context.client_context: object
         Additional context when invoked through AWS Mobile SDK
    context.function_name: str
         Lambda function name
    context.function_version: str
         Function version identifier
    context.get_remaining_time_in_millis: function
         Time in milliseconds before function times out
    context.identity:
         Cognito identity provider context when invoked through AWS Mobile SDK
    context.invoked_function_arn: str
         Function ARN
    context.log_group_name: str
         Cloudwatch Log group name
    context.log_stream_name: str
         Cloudwatch Log stream name
    context.memory_limit_in_mb: int
        Function memory

        https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict
        'statusCode' and 'body' are required

        {
            "isBase64Encoded": true | false,
            "statusCode": httpStatusCode,
            "headers": {"headerName": "headerValue", ...},
            "body": "..."
        }

        # api-gateway-simple-proxy-for-lambda-output-format
        https: // docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """



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
        "headers": { 
            "Access-Control-Allow-Origin": "*" 
        },
        'body': json.dumps('Successfully Submitted!')
    }
