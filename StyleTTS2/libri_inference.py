import torch
import random
import numpy as np
import time
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textcleaner = TextCleaner()
from phonemizer.backend import EspeakBackend
from Utils.PLBERT.util import load_plbert 
from collections import OrderedDict
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import soundfile as sf

class StyleTTS2Inference:
    def __init__(self, model, config_path="StyleTTS2/LibriTTS/config.yml"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = yaml.safe_load(open(config_path))
        self.model_params = Munch(self.config['model_params'])
        self.global_phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4
    
    def load_models(self, models_path):
        text_aligner = load_ASR_models("Utils/ASR/config.yml", "Utils/ASR/epoch_00080.pth")
        pitch_extractor = load_F0_models(self.config['F0_path'])
        plbert = load_plbert(self.config['PLBERT_dir'])
    
        # FULL NESTED CONVERSION:
        from munch import munchify
        model_params = munchify(self.config['model_params'])
    
        if isinstance(model_params, dict):
            model_params = Munch(model_params)
            print("DEBUG model_params.decoder:", model_params.get('decoder', 'MISSING'))
        if isinstance(model_params, Munch) and 'decoder' in model_params:
            print("DEBUG model_params.decoder.type:", getattr(model_params.decoder, 'type', 'MISSING'))
    
        self.model = build_model(model_params, text_aligner, pitch_extractor, plbert)


    
        # Load weights
        params_whole = torch.load(models_path, map_location='cpu', weights_only=False)
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        
        for key in self.model: 
            self.model[key].eval()
            self.model[key].to(self.device)
        
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )
   
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        textcleaner = TextCleaner()
        tokens = textcleaner(ps)
        tokens.insert(0, 0)

        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device), 
                                embedding=bert_dur,
                                embedding_scale=embedding_scale,
                                features=ref_s, # reference from the same speaker as the embedding
                                num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            
            d = self.model.predictor.text_encoder(d_en, 
                                                s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(dim=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if True:  # Using loaded model
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model['predictor'].F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if True:
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model['decoder'](asr, 
                                      F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            
        return out.squeeze().cpu().numpy()[..., :-50]


if __name__ == "__main__":
    synthesizer = StyleTTS2Inference(
        model={},
        config_path = "C:/elevenlabs-clone/StyleTTS2/LibriTTS/config.yml"
    )
    synthesizer.load_models("C:/elevenlabs-clone/StyleTTS2/LibriTTS/epochs_2nd_00020.pth")

    ref_style_path = synthesizer.compute_style("C:/elevenlabs-clone/StyleTTS2/LibriTTS/woman1.wav")

    # GENERATE AUDIO
    text = "Hello, welcome to our text to speech synthesis demo"
    audio = synthesizer.inference(
        text,
        ref_s = ref_style_path
    )


    sf.write("output.wav", audio, 24000)
    