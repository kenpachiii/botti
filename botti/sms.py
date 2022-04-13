import datetime

import boto3
from botocore.exceptions import ClientError

import logging

logger = logging.getLogger(__name__)

# https://aws.amazon.com/premiumsupport/knowledge-center/ec2-port-25-throttle/

def exception(client, recipients: list, body: str):

    return

    body = '{time} - {body}'.format(time=datetime.datetime.utcnow(), body = body)

    return client.send_email(
        Destination={
            'ToAddresses': recipients,
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': body,
                },
            },
        },
        Source='Sender Name <hiddenleafresearch@gmail.com>',
    )
    
def profits(client, recipients: list, body: str):
    return
    return client.send_email(
        Destination={
            'ToAddresses': recipients,
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': body,
                },
            },
        },
        Source='Sender Name <hiddenleafresearch@gmail.com>',
    )

def send_sms(type: str, msg: str) -> None:

    client = boto3.client('ses', region_name='us-east-1')

    try:
        if type == 'exception':
            exception(client, ['9286323030@vtext.com'], msg)

        if type == 'profits':
            profits(client, ['3868372377@txt.att.net', '9286323030@vtext.com'], msg)

    except ClientError as e:
        logger.error('error sending sms {}'.format(str(e)))

    

