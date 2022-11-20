from NLP_Models import deepHateSpeechDetection as dhsd
import logging
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi. Please enter your keywords!')


def nlpResult(update: Update, context: CallbackContext):
    hate = dhsd.hateSpeechPredict(update.message.text)
    result = {'result':hate['final_result'], 'confidence': hate['confidence']}    
    update.message.reply_text('This is the result of' + '\t' + str(update.message.text) +': ' +'\t' +str(result))
    
    #return result
    

def main():
    updater = Updater('1891956699:AAEd5fl8bKxoa8H3NtPt_ANByeQ1GQuhCTs')
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, nlpResult))
    updater.start_polling()
    updater.idle()
if __name__ == '__main__':
    main()
