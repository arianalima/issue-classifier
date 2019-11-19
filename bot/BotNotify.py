import sys

import telegram
import configparser

# Configuring bot
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))
token = config['DEFAULT']['token']
support_chat_id = config['DEFAULT']['support_chat_id']

def notify(message):
    bot = telegram.Bot(token)
    bot.sendMessage(chat_id=sys.argv[2], text=message)
