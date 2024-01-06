# https://yagmail.readthedocs.io/en/latest/usage.html
import yagmail
from pathlib import Path
import os
import json
from platform import node
# your machine will call you back if your super duper ML model is trained and skynet is ready to take over the world
parent_dir = Path(__file__).resolve().parent
nonpublic_dir =str(Path.joinpath(parent_dir, "nonpublic")) + os.sep
my_creds_filename = "usercredentials.json"


with open(nonpublic_dir+my_creds_filename) as json_file:
    creds_data = json.load(json_file)


def send_finish_notification():
    MACHINE_ID = node()
    key_filename = creds_data["my_credfile"]
    yag = yagmail.SMTP(user=creds_data["my_email"], oauth2_file=nonpublic_dir+key_filename)
    yag.send(to=creds_data["target_email"], contents=MACHINE_ID + ": Job done.", subject = MACHINE_ID + ': Python done fine')

def send_error_notification(errorstring):
    MACHINE_ID = node()
    key_filename = creds_data["my_credfile"]
    yag = yagmail.SMTP(user=creds_data["my_email"], oauth2_file=nonpublic_dir+key_filename)
    yag.send(to=creds_data["target_email"], contents=str(errorstring), subject = MACHINE_ID + ": Job failed. Go fix it, human!")

