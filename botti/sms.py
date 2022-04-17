import logging
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

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

        with smtplib.SMTP("smtp.gmail.com", 587) as server:

            server.ehlo()
            server.starttls()
            server.login('botti.notification@gmail.com', 'yygakfowwmpogiuy')
            
            recipients = None
            if type == 'exception':
                recipients = ['9286323030@vtext.com']

            if type == 'profits':
                recipients = ['9286323030@vtext.com'] # '3868372377@txt.att.net'

            msg = MIMEText(msg)
            msg['From'] = 'botti.notification@gmail.com'
            msg['To'] = ', '.join(recipients)

            server.sendmail('botti.notification@gmail.com', recipients, msg.as_string())

            server.close()

    except Exception as e:
        logger.error('error sending sms {}'.format(str(e)))

    