# Email configuration (here for Gmail account as sender mail)
smtp_server = 'smtp.gmail.com'  # SMTP server address
smtp_port = 587  # SMTP port (587 for TLS)

smtp_password = 'ctij wnin frxl ggup'
sender_email = 'maexemc@gmail.com'
recipient_emails = ['maexemc@gmail.com']

def get_email_subject(class_name, time, manual):

    if not manual:
        return f"Husky: Automatic Object Detection of '{class_name}'"
    else:
        class_name = class_name.replace("-", ", ")
        if class_name == "":
            class_name = "nothing"
        return f"Husky: Manual Object Detection of '{class_name}'"

def get_email_body(class_name, time, manual):
    

    if "-" in class_name:
        object_text = "The objects"
        have_text = "have"
        # Replace "-" with ", " in the class_name
        class_name = class_name.replace("-", ", ")
        class_name = f"'{class_name}' "
    elif class_name == "":
        object_text = "No objects"
        have_text = "have"
    else:
        object_text = "The object"
        have_text = "has"
        class_name = f"'{class_name}' "

    if not manual:
        
        return f"""Dear User,

{object_text} {class_name}{have_text} been detected automatically at time: {time}
Kind Regards, Alascom
"""
    else:
        return f"""Dear User,

{object_text} {class_name}{have_text} been detected during the manual recording at time: {time}
Kind Regards, Alascom
"""