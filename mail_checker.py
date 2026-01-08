import imaplib
import email

# Gmail credentials
user = 'Your Gmail address'
password = 'Your password'

# Connect to Gmail IMAP
mail = imaplib.IMAP4_SSL('imap.gmail.com')
mail.login(user, password)
mail.list()
mail.select("inbox")

def readMails(address):
    # Search for emails from address
    result, data = mail.search(None, '(FROM "' + address + '")')
    ids = data[0]
    id_list = ids.split()
    latest_email_id = id_list[-1]
    
    # Fetch and parse email
    result, data = mail.fetch(latest_email_id, "(RFC822)")
    raw_email = data[0][1]
    email_message = email.message_from_bytes(raw_email)
    subject = email_message['Subject']
    
    # Print details
    print('-------FROM ' + address + '-------\n\n')
    print("Subject: ", subject, '\n\n')

listOfMails = ['mail1@gmail.com', 'mail2@gmail.com']
for addresses in listOfMails:
    readMails(addresses)
