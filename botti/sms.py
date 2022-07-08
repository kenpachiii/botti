import logging
import smtplib
import time
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

queue = {}

def program_time() -> int:
    return int(time.time() * 1000)

def delay_message(msg: str) -> bool:
    return msg in queue.keys() and (program_time() - queue[msg]) < 3600000

def update_queue(msg: str) -> None:

    if msg in queue.keys():
        return

    queue[msg] = program_time()

def exception(server: smtplib.SMTP, recipients: list, msg: str):
    msg = MIMEText(msg)
    msg['From'] = 'botti.notification@gmail.com'
    msg['To'] = ', '.join(recipients)

    server.sendmail('botti.notification@gmail.com', recipients, msg.as_string())
    
def profits(server: smtplib.SMTP, recipients: list, msg: str):
    msg = MIMEText(msg)
    msg['From'] = 'botti.notification@gmail.com'
    msg['To'] = ', '.join(recipients)

    server.sendmail('botti.notification@gmail.com', recipients, msg.as_string())

def send_sms(type: str, msg: str) -> None:
    try:

        if not delay_message(msg):
            with smtplib.SMTP("smtp.gmail.com", 587) as server:

                server.ehlo()
                server.starttls()
                server.login('botti.notification@gmail.com', 'yygakfowwmpogiuy')
                
                recipients = None
                if type == 'exception' or type == 'system-status':
                    recipients = ['9286323030@vtext.com']

                if type == 'profits':
                    recipients = ['9286323030@vtext.com', '3868372377@txt.att.net'] 

                msg = MIMEText(msg)
                msg['From'] = 'botti.notification@gmail.com'
                msg['To'] = ', '.join(recipients)

                server.sendmail('botti.notification@gmail.com', recipients, msg.as_string())
                server.close()

                update_queue(msg)

    except Exception as e:
        logger.error('failed sending sms {}'.format(str(e)))    