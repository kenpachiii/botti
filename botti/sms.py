import logging
import smtplib
import time
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

queue = []

def delay_message(msg: str) -> bool:

    delay = False

    for i in range(0, len(queue)):

        (sms, timestamp) = queue[i]
        if msg == sms and int(time.time() * 1000) - timestamp < 3600:
            delay = True

    return delay

def update_queue(msg: str) -> None:
    
    for i in range(0, len(queue)):

        (sms, timestamp) = queue[i]
        if msg == sms:
            queue.pop(i)
            return

    queue.append((msg, int(time.time() * 1000)))

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
                if type == 'exception' or type == 'earlyexit' or type == 'system-status':
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
        pass

    