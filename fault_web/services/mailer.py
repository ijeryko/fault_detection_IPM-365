import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class GmailMailer:
    def __init__(self, gmail_address: str, app_password: str, from_name: str):
        self.gmail_address = gmail_address
        self.app_password = app_password
        self.from_name = from_name

    def is_configured(self) -> bool:
        return bool(self.gmail_address and self.app_password)

    def send_email(self, to_emails: list[str], subject: str, body_text: str) -> None:
        if not self.is_configured():
            raise RuntimeError("Gmail not configured.")

        msg = MIMEMultipart()
        msg["From"] = f"{self.from_name} <{self.gmail_address}>"
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(self.gmail_address, self.app_password)
            server.sendmail(self.gmail_address, to_emails, msg.as_string())