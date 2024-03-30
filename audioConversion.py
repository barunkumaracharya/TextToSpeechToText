import whisper
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

## Convert from audio to text
model = whisper.load_model("base")
result = model.transcribe('Sample2.m4a')
print(result["text"])


## Convert from text to audioff
ckpt_base = 'D:\\github\\trystWithAi\\OpenVoice\\checkpoints\\base_speakers\\EN'
ckpt_converter = 'D:\\github\\trystWithAi\\OpenVoice\\checkpoints\\converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'D:\\github\\trystWithAi\\OpenVoice\\outputs'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}\\config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}\\checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}\\config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}\\checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
source_se = torch.load(f'{ckpt_base}\\en_default_se.pth').to(device)
reference_speaker = 'D:\\github\\trystWithAi\\OpenVoice\\resources\\demo_bka_speaker.mp3'
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)
save_path = f'{output_dir}\\output_train.wav'

# Run the base speaker tts
text = result['text']
src_path = f'{output_dir}\\tmp.wav'
base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message)


# audio = whisper.load_audio("Sample1.mp")git clone https://github.com/myshell-ai/OpenVoice.git