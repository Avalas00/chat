

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qq -U diffusers datasets transformers accelerate ftfy pyarrow==9.0.0

from huggingface_hub import notebook_login

notebook_login()

!pip install transformers
!pip install torch
!pip install librosa
!pip install soundfile
!pip install pydub
!pip install langchain
!pip install gtts

import torch
device = torch.device("cuda" if torch.cuda.is_availabe() else "cpu")

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

# 加載Wav2Vec2模型和分詞器
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加載語音數據
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 將語音數據轉換為文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = tokenizer(speech, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# 示例
file_path = "test.wav"
text = speech_to_text(file_path)
print("Transcription: ", text)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加載GPT-2模型和分詞器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 使用LLM生成回應
def generate_response(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例
response = generate_response(text)
print("Response: ", response)

from gtts import gTTS
import os

def text2speech(text, output_file):
  tts = gTTS(text=text, lang='en')
  tts.save(output_file)

output_file = "response.mp3"
text2speech("Aduio response saved to ", output_file)
os.system(f"start{output_file}")

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2模型和分词器
wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载GPT-2模型和分词器
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = wav2vec_tokenizer(speech, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec_tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# 使用LLM生成响应
def generate_response(text):
    system_prompt = (
        "You are an AI language model trained to assist with various tasks. Please generate a helpful and coherent response based on the following input.\n\n"
        "User input: "
    )
    prompt = system_prompt + text
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7, top_p=0.9)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用LLM生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "test1.wav"  # 请上传你的音频文件并使用该文件路径
audio = main(audio_file_path)
audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2处理器和模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载GPT-2模型和分词器
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = processor(speech, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 使用LLM生成响应（使用改进的提示）
def generate_response(text):
    prompt = f"The user said: {text}\nAI response:"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=gpt2_tokenizer.eos_token_id)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用LLM生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "test1.wav"  # 请上传你的音频文件并使用该文件路径
audio = main(audio_file_path)
audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2处理器和模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载GPT-2模型和分词器
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 使用LLM生成响应（使用改进的提示）
def generate_response(text):
    prompt = f"The user said: {text}\nAI response:"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用LLM生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "joke.wav"
audio = main(audio_file_path)
audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2处理器和模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载Mixtral模型和分词器
mixtral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
mixtral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 使用Mixtral模型生成响应
def generate_response(text):
    prompt = f"The user said: {text}\nAI response:"
    inputs = mixtral_tokenizer.encode(prompt, return_tensors="pt")
    outputs = mixtral_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=mixtral_tokenizer.eos_token_id
    )
    response = mixtral_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用Mixtral生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "joke.wav"
audio = main(audio_file_path)
audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2处理器和模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载Mixtral模型和分词器
mixtral_tokenizer = AutoTokenizer.from_pretrained("Enagamirzayev/whisper-small-llm-lingo_v")
mixtral_model = AutoModelForCausalLM.from_pretrained("Enagamirzayev/whisper-small-llm-lingo_v")

# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 使用Mixtral模型生成响应
def generate_response(text):
    prompt = f"The user said: {text}\nAI response:"
    inputs = mixtral_tokenizer.encode(prompt, return_tensors="pt")
    outputs = mixtral_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=mixtral_tokenizer.eos_token_id
    )
    response = mixtral_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用Mixtral生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "joke.wav"
audio = main(audio_file_path)
audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import librosa
import IPython.display as ipd

# 加载Wav2Vec2处理器和模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载Mixtral模型和分词器
# mixtral_tokenizer = AutoTokenizer.from_pretrained("omi-health/sum-small")
# mixtral_model = AutoModelForCausalLM.from_pretrained("omi-health/sum-small")
# mixtral_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# mixtral_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

mixtral_tokenizer = AutoTokenizer.from_pretrained("Cossale/tinyllama-claude_16bit")
mixtral_model = AutoModelForCausalLM.from_pretrained("Cossale/tinyllama-claude_16bit")
# 加载音频数据
def load_audio(file_path):
    speech, rate = librosa.load(file_path, sr=16000)
    return speech

# 语音转文本
def speech_to_text(file_path):
    speech = load_audio(file_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 使用Mixtral模型生成响应
def generate_response(text):
    prompt = f"The user said: {text}\nAI response:"
    inputs = mixtral_tokenizer.encode(prompt, return_tensors="pt")
    outputs = mixtral_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=mixtral_tokenizer.eos_token_id
    )
    response = mixtral_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 文本转语音
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# 主程序
def main(audio_file_path):
    # 语音转文本
    text = speech_to_text(audio_file_path)
    print("Transcription: ", text)

    # 使用Mixtral生成响应
    response = generate_response(text)
    print("Response: ", response)

    # 文本转语音
    output_file = "response.mp3"
    text_to_speech(response, output_file)
    print("Audio response saved to ", output_file)

    # 使用IPython显示和播放音频
    return ipd.Audio(output_file)

# 示例
audio_file_path = "test1.wav"
audio = main(audio_file_path)
audio