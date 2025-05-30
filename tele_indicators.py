from time import sleep

class TypingIndicator:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self._stop_event = False
        
    def run(self):
        while not self._stop_event:
            try:
                self.bot.send_chat_action(self.chat_id, 'typing')
                sleep(2.5)  # Send every 2.5 seconds (Telegram caches for 5s)
            except:
                break
                
    def stop(self):
        self._stop_event = True

class SendingPhotoIndicator:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self._stop_event = False
        
    def run(self):
        while not self._stop_event:
            try:
                self.bot.send_chat_action(self.chat_id, 'upload_photo')
                sleep(2.5)  # Send every 2.5 seconds (Telegram caches for 5s)
            except:
                break
                
    def stop(self):
        self._stop_event = True