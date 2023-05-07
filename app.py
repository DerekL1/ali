from flask import Flask, render_template, request
app = Flask(__name__)

import whisper
import translators as ts

@app.route('/runModel', methods=['GET', 'POST'])
def run_whisper():
    # gets jsdata from ajax post method and saves it locally
    jsdata = request.files['audioId']
    path='./audioFile.wav'
    jsdata.save(path)
    whisper_model = whisper.load_model("base")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    detLang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    # return the detected language and transcription
    return {"language": detLang, "transcription": result.text}

@app.route('/runTranslation', methods=['GET', 'POST'])
def run_translation():
    #get old language, new language, and transcription
    ogLang = request.form['oglang']
    newLang = request.form['newlang']
    transcription = request.form['transcript']
    translation=transcription
    #run translation api on it
    if ogLang!=newLang:
        translation = ts.translate_text(transcription, translator='google',from_language=ogLang, to_language=newLang)
    return translation

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')
        
if __name__ == "__main__":
    app.run(debug=True)