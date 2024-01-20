from gtts import gTTS
#from playsound import playsound
import os


def read_string(msg):
    language = 'pl'
    speech = gTTS(text=msg, lang=language, slow=False)
    speech.save("text.mp3")
   # playsound("text.mp3")
    print("sound")
    os.system('mpg321 text.mp3 &')
