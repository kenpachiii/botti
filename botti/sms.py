import datetime
import logging
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

def exception(server: smtplib.SMTP, recipients: list, msg: str):
    msg = MIMEText('time: {time}\n\n{msg}'.format(time=datetime.datetime.utcnow().strftime('%H:%M:%S'), msg = msg))
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
            
            if type == 'exception':
                exception(server, ['9286323030@vtext.com'], msg)

            if type == 'profits':
                profits(server, ['9286323030@vtext.com'], msg) # '3868372377@txt.att.net'

            server.close()

    except Exception as e:
        logger.error('error sending sms {}'.format(str(e)))

    