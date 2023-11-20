# Email configuration (here for Gmail account as sender mail)
smtp_server = 'smtp.gmail.com'  # SMTP server address
smtp_port = 587  # SMTP port (587 for TLS)

smtp_password = 'ctij wnin frxl ggup'
sender_email = 'maexemc@gmail.com'
recipient_emails = ['maexemc@gmail.com']

def get_email_subject(class_name, time):
    
    return f"Husky: Object Detection of '{class_name}'"


def get_email_body(class_name, time):
    
    return f"""Ciao Bello,

The object '{class_name}' has been detected at time: {time}
Cheers, Massimo
"""
