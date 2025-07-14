import os
import sys
import argparse
from pathlib import Path
import torch
import librosa
import numpy as np
import soundfile as sf
import yaml
import hashlib
import warnings
import gc
import json
import math
import time
from ml_collections import ConfigDict

# Add correct paths for vendored code
current_dir = Path(__file__).parent.absolute()
sys.path.append(os.path.join(current_dir, 'ultimatevocalremovergui', 'lib_v5'))

from lib_v5 import spec_utils
from lib_v5.vr_network import nets, nets_new
from lib_v5.vr_network.model_param_init import ModelParameters
from lib_v5.tfc_tdf_v3 import TFC_TDF_net
from scipy.io import wavfile
from types import SimpleNamespace
from gui_data.constants import *

warnings.filterwarnings("ignore")

from types import SimpleNamespace
import pydub

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_format(audio_path, save_format, mp3_bit_set):
    
    if not save_format == WAV:
        
        if OPERATING_SYSTEM == 'Darwin':
            FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
            pydub.AudioSegment.converter = FFMPEG_PATH
        
        musfile = pydub.AudioSegment.from_wav(audio_path)
        
        if save_format == FLAC:
            audio_path_flac = audio_path.replace(".wav", ".flac")
            musfile.export(audio_path_flac, format="flac")  
        
        if save_format == MP3:
            audio_path_mp3 = audio_path.replace(".wav", ".mp3")
            try:
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set, codec="libmp3lame")
            except Exception as e:
                print(e)
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set)
        
        try:
            os.remove(audio_path)
        except Exception as e:
            print(e)

# a minimal var stub with get()/set()
class DummyVar:
    def __init__(self, value=None):
        self._value = value
    def get(self):
        return self._value
    def set(self, v):
        self._value = v

# build a headless `root` from DEFAULT_DATA:
root = SimpleNamespace()
for key, val in DEFAULT_DATA.items():
    # e.g. DEFAULT_DATA['window_size']=$512 ⇒ root.window_size_var = DummyVar(512)
    setattr(root, f"{key}_var", DummyVar(val))


root.mdxnet_stems_var = DummyVar(VOCAL_STEM)

# any true “plain” attributes your code reads directly:
# (search your code for `root.wav_type_set =` or `root.save_format =` etc)
root.wav_type_set = DEFAULT_DATA['wav_type_set']
root.save_format   = DEFAULT_DATA['save_format']
root.device_set    = DEFAULT_DATA['device_set']
# ——— stubs for all the other root.X references ———




# --- CONSTANTS ---
#Models
BASE_PATH = current_dir
MODELS_DIR = os.path.join(BASE_PATH, 'ultimatevocalremovergui', 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'ultimatevocalremovergui', 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')



# Attempt to load each one, fall back to empty dict if it’s missing/corrupt
def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''
    with open(dictionary, 'r') as d:
        return json.load(d)

VR_HASH_MAPPER            = load_model_hash_data(VR_HASH_JSON)
MDX_HASH_MAPPER           = load_model_hash_data(MDX_HASH_JSON)
MDX_NAME_SELECT_MAPPER    = load_model_hash_data(MDX_MODEL_NAME_SELECT)
DEMUCS_NAME_SELECT_MAPPER = load_model_hash_data(DEMUCS_MODEL_NAME_SELECT)

# 1) any mapping dicts you import from gui_data.constants:
root.vr_hash_MAPPER       = VR_HASH_MAPPER
root.mdx_hash_MAPPER      = MDX_HASH_MAPPER
root.mdx_name_select_MAPPER   = MDX_NAME_SELECT_MAPPER
root.demucs_name_select_MAPPER= DEMUCS_NAME_SELECT_MAPPER

# 2) any methods ModelData calls on root:
root.return_ensemble_stems                = lambda: (VOCAL_STEM, INST_STEM)
root.check_only_selection_stem            = lambda stem: False
root.process_determine_secondary_model    = lambda *args, **kwargs: (None, None)
root.process_determine_vocal_split_model  = lambda: None
root.process_determine_demucs_pre_proc_model = lambda stem: None

# 3) the two pop-up hooks that get_model_data_from_popup expects:
root.pop_up_vr_param  = lambda model_hash: None
root.pop_up_mdx_model = lambda model_hash, model_path: None

# 4) the results those pop-ups are supposed to write into:
root.vr_model_params  = {}
root.mdx_model_params = {}
model_hash_table = {}

# --- HELPER FUNCTIONS ---

def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def secondary_stem(primary_stem):
    return INST_STEM if primary_stem == VOCAL_STEM else VOCAL_STEM

def prepare_mix(mix):
    
    audio_path = mix
    
    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if isinstance(audio_path, str):
        if not np.any(mix) and audio_path.endswith('.mp3'):
            mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix


def loading_mix(X, mp):

    X_wave, X_spec_s = {}, {}
    
    bands_n = len(mp.param['band'])
    
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
    
        if OPERATING_SYSTEM == 'Darwin':
            wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
        else:
            wav_resolution = 'polyphase'#bp['res_type']
    
        if d == bands_n: # high-end band
            X_wave[d] = X

        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution) # "kaiser_best"
            
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=True)
        
        # if d == bands_n and is_high_end_process:
        #     input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
        #     input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s

    return X_spec

def vr_denoiser(X, device, hop_length=1024, n_fft=2048, cropsize=256, is_deverber=False, model_path=None):
    batchsize = 4

    if is_deverber:
        nout, nout_lstm = 64, 128
        mp = ModelParameters(os.path.join(BASE_PATH,  'ultimatevocalremovergui', 'lib_v5', 'vr_network', 'modelparams', '4band_v3.json'))
        n_fft = mp.param['bins'] * 2
    else:
        mp = None
        hop_length=1024
        nout, nout_lstm = 16, 128
    # breakpoint()
    model = nets_new.CascadedNet(n_fft, nout=nout, nout_lstm=nout_lstm)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(device)

    if mp is None:
        X_spec = spec_utils.wave_to_spectrogram_old(X, hop_length, n_fft)
    else:
        X_spec = loading_mix(X.T, mp)
   
    #PreProcess
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    #Sep
    n_frame = X_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, cropsize, model.offset)
    X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    X_mag_pad /= X_mag_pad.max()

    X_dataset = []
    patches = (X_mag_pad.shape[2] - 2 * model.offset) // roi_size
    for i in range(patches):
        start = i * roi_size
        X_mag_crop = X_mag_pad[:, :, start:start + cropsize]
        X_dataset.append(X_mag_crop)

    X_dataset = np.asarray(X_dataset)

    model.eval()
    
    with torch.no_grad():
        mask = []
        # To reduce the overhead, dataloader is not used.
        for i in range(0, patches, batchsize):
            X_batch = X_dataset[i: i + batchsize]
            X_batch = torch.from_numpy(X_batch).to(device)

            pred = model.predict_mask(X_batch)

            pred = pred.detach().cpu().numpy()
            pred = np.concatenate(pred, axis=2)
            mask.append(pred)

        mask = np.concatenate(mask, axis=2)
    
    mask = mask[:, :, :n_frame]

    #Post Proc
    if is_deverber:
        v_spec = mask * X_mag * np.exp(1.j * X_phase)
        y_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
    else:
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

    if mp is None:
        wave = spec_utils.spectrogram_to_wave_old(v_spec, hop_length=1024)
    else:
        wave = cmb_spectrogram_to_wave(v_spec, mp, is_v51_model=True).T
        
    wave = spec_utils.match_array_shapes(wave, X)

    if is_deverber:
        wave_2 = cmb_spectrogram_to_wave(y_spec, mp, is_v51_model=True).T
        wave_2 = spec_utils.match_array_shapes(wave_2, X)
        return wave, wave_2
    else:
        return wave


# --- CLASSES ---
DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False):

        device_set = root.device_set_var.get()
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = root.is_deverb_vocals_var.get() if os.path.isfile(DEVERBER_MODEL_PATH) else False
        self.deverb_vocal_opt = DEVERB_MAPPER[root.deverb_vocal_opt_var.get()]
        self.is_denoise_model = True if root.denoise_option_var.get() == DENOISE_M and os.path.isfile(DENOISER_MODEL_PATH) else False
        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1
        self.is_normalization = root.is_normalization_var.get()#
        self.is_use_opencl = False#True if is_opencl_only else root.is_use_opencl_var.get()
        self.is_primary_stem_only = root.is_primary_stem_only_var.get()
        self.is_secondary_stem_only = root.is_secondary_stem_only_var.get()
        self.is_denoise = True if not root.denoise_option_var.get() == DENOISE_NONE else False
        self.is_mdx_c_seg_def = root.is_mdx_c_seg_def_var.get()#
        self.mdx_batch_size = 1 if root.mdx_batch_size_var.get() == DEF_OPT else int(root.mdx_batch_size_var.get())
        self.mdxnet_stem_select = root.mdxnet_stems_var.get() 
        self.overlap = float(root.overlap_var.get()) if not root.overlap_var.get() == DEFAULT else 0.25
        self.overlap_mdx = float(root.overlap_mdx_var.get()) if not root.overlap_mdx_var.get() == DEFAULT else root.overlap_mdx_var.get()
        self.overlap_mdx23 = int(float(root.overlap_mdx23_var.get()))
        self.semitone_shift = float(root.semitone_shift_var.get())
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = root.is_match_frequency_pitch_var.get()
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = root.is_mdx23_combine_stems_var.get()#
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = root.wav_type_set#
        self.device_set = device_set.split(':')[-1].strip() if ':' in device_set else device_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        self.is_invert_spec = root.is_invert_spec_var.get()#
        self.is_mixer_mode = False#
        self.demucs_stems = root.demucs_stems_var.get()
        self.is_demucs_combine_stems = root.is_demucs_combine_stems_var.get()
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = root.is_save_inst_set_vocal_splitter_var.get()
        self.is_inst_only_voc_splitter = root.check_only_selection_stem(INST_STEM_ONLY)
        self.is_save_vocal_only = root.check_only_selection_stem(IS_SAVE_VOC_ONLY)

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(ENSEMBLE_PARTITION)
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            if root.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif root.ensemble_main_stem_var.get() == MULTI_STEM_ENSEMBLE and root.chosen_process_method_var.get() == ENSEMBLE_MODE:
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if is_not_vocal_stem else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.aggression_setting = float(int(root.aggression_setting_var.get())/100)
            self.is_tta = root.is_tta_var.get()
            self.is_post_process = root.is_post_process_var.get()
            self.window_size = int(root.window_size_var.get())
            self.batch_size = 1 if root.batch_size_var.get() == DEF_OPT else int(root.batch_size_var.get())
            self.crop_size = int(root.crop_size_var.get())
            self.is_high_end_process = 'mirroring' if root.is_high_end_process_var.get() else 'None'
            self.post_process_threshold = float(root.post_process_threshold_var.get())
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            # breakpoint()
            if self.model_hash:
                self.model_hash_dir = os.path.join(VR_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = secondary_stem(self.primary_stem)
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    self.primary_stem_native = self.primary_stem
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()
   
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.margin = int(root.margin_var.get())
            self.chunks = 0
            self.mdx_segment_size = int(root.mdx_segment_size_var.get())
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

                            self.mdx_c_configs = config
                                
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if root.compensate_var.get() == AUTO_SELECT else float(root.compensate_var.get())
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                        
                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var.get() if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not root.demucs_stems_var.get() in [VOCAL_STEM, INST_STEM] else False
            self.margin_demucs = int(root.margin_demucs_var.get())
            self.chunks_demucs = 0
            self.shifts = int(root.shifts_var.get())
            self.is_split_mode = root.is_split_mode_var.get()
            self.segment = root.segment_var.get()
            self.is_chunk_demucs = root.is_chunk_demucs_var.get()
            self.is_primary_stem_only = root.is_primary_stem_only_var.get() if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var.get() 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var.get() if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var.get()
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = root.demucs_stems_var.get() == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var.get() if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)
            
        self.vocal_splitter_model_data()
            
    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = root.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False
            
            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True
            
    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
            
        #print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)
              
    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]#
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]#
   
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = secondary_stem(self.primary_stem)
            
    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return self.get_model_data_from_popup()

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        if self.is_dry_check:
            return None
            
        if self.process_method == VR_ARCH_TYPE:
            root.pop_up_vr_param(self.model_hash)
            return root.vr_model_params
        elif self.process_method == MDX_ARCH_TYPE:
            root.pop_up_mdx_model(self.model_hash, self.model_path)
            return root.mdx_model_params

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)
                
        #print(self.model_name," - ", self.model_hash)

class SeperateAttributes:
    def __init__(self, model_data: ModelData, 
                 process_data: dict, 
                 main_model_primary_stem_4_stem=None, 
                 main_process_method=None, 
                 is_return_dual=True, 
                 main_model_primary=None, 
                 vocal_stem_path=None, 
                 master_inst_source=None,
                 master_vocal_source=None):
        
        self.list_all_models: list
        self.process_data = process_data
        self.progress_value = 0
        self.set_progress_bar = process_data['set_progress_bar']
        if vocal_stem_path:
            self.audio_file, self.audio_file_base = vocal_stem_path
            self.audio_file_base_voc_split = lambda stem, split:os.path.join(self.export_path, f'{self.audio_file_base.replace("_(Vocals)", "")}_({stem}_{split}).wav')
        else:
            self.audio_file = process_data['audio_file']
            self.audio_file_base = process_data['audio_file_base']
            self.audio_file_base_voc_split = None
        self.export_path = process_data['export_path']
        self.cached_source_callback = process_data['cached_source_callback']
        self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']
        self.list_all_models = process_data['list_all_models']
        self.process_iteration = process_data['process_iteration']
        self.is_return_dual = is_return_dual
        self.is_pitch_change = model_data.is_pitch_change
        self.semitone_shift = model_data.semitone_shift
        self.is_match_frequency_pitch = model_data.is_match_frequency_pitch
        self.overlap = model_data.overlap
        self.overlap_mdx = model_data.overlap_mdx
        self.overlap_mdx23 = model_data.overlap_mdx23
        self.is_mdx_combine_stems = model_data.is_mdx_combine_stems
        self.is_mdx_c = model_data.is_mdx_c
        self.mdx_c_configs = model_data.mdx_c_configs
        self.mdxnet_stem_select = model_data.mdxnet_stem_select
        self.mixer_path = model_data.mixer_path
        self.model_samplerate = model_data.model_samplerate
        self.model_capacity = model_data.model_capacity
        self.is_vr_51_model = model_data.is_vr_51_model
        self.is_pre_proc_model = model_data.is_pre_proc_model
        self.is_secondary_model_activated = model_data.is_secondary_model_activated if not self.is_pre_proc_model else False
        self.is_secondary_model = model_data.is_secondary_model if not self.is_pre_proc_model else True
        self.process_method = model_data.process_method
        self.model_path = model_data.model_path
        self.model_name = model_data.model_name
        self.model_basename = model_data.model_basename
        self.wav_type_set = model_data.wav_type_set
        self.mp3_bit_set = model_data.mp3_bit_set
        self.save_format = model_data.save_format
        self.is_gpu_conversion = model_data.is_gpu_conversion
        self.is_normalization = model_data.is_normalization
        self.is_primary_stem_only = model_data.is_primary_stem_only if not self.is_secondary_model else model_data.is_primary_model_primary_stem_only
        self.is_secondary_stem_only = model_data.is_secondary_stem_only if not self.is_secondary_model else model_data.is_primary_model_secondary_stem_only      
        self.is_ensemble_mode = model_data.is_ensemble_mode
        self.secondary_model = model_data.secondary_model #
        self.primary_model_primary_stem = model_data.primary_model_primary_stem
        self.primary_stem_native = model_data.primary_stem_native
        self.primary_stem = model_data.primary_stem #
        self.secondary_stem = model_data.secondary_stem #
        self.is_invert_spec = model_data.is_invert_spec #
        self.is_deverb_vocals = model_data.is_deverb_vocals
        self.is_mixer_mode = model_data.is_mixer_mode #
        self.secondary_model_scale = model_data.secondary_model_scale #
        self.is_demucs_pre_proc_model_inst_mix = model_data.is_demucs_pre_proc_model_inst_mix #
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None
        self.main_model_primary_stem_4_stem = main_model_primary_stem_4_stem
        self.main_model_primary = main_model_primary
        self.ensemble_primary_stem = model_data.ensemble_primary_stem
        self.is_multi_stem_ensemble = model_data.is_multi_stem_ensemble
        self.is_other_gpu = False
        self.is_deverb = True
        self.DENOISER_MODEL = model_data.DENOISER_MODEL
        self.DEVERBER_MODEL = model_data.DEVERBER_MODEL
        self.is_source_swap = False
        self.vocal_split_model = model_data.vocal_split_model
        self.is_vocal_split_model = model_data.is_vocal_split_model
        self.master_vocal_path = None
        self.set_master_inst_source = None
        self.master_inst_source = master_inst_source
        self.master_vocal_source = master_vocal_source
        self.is_save_inst_vocal_splitter = isinstance(master_inst_source, np.ndarray) and model_data.is_save_inst_vocal_splitter
        self.is_inst_only_voc_splitter = model_data.is_inst_only_voc_splitter
        self.is_karaoke = model_data.is_karaoke
        self.is_bv_model = model_data.is_bv_model
        self.is_bv_model_rebalenced = model_data.bv_model_rebalance and self.is_vocal_split_model
        self.is_sec_bv_rebalance = model_data.is_sec_bv_rebalance
        self.stem_path_init = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
        self.deverb_vocal_opt = model_data.deverb_vocal_opt
        self.is_save_vocal_only = model_data.is_save_vocal_only
        self.device = DEVICE
        self.run_type = ['CPUExecutionProvider']
        self.is_opencl = False
        self.device_set = model_data.device_set
        self.is_use_opencl = model_data.is_use_opencl
        
        if self.is_inst_only_voc_splitter or self.is_sec_bv_rebalance:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
        
        if main_model_primary and self.is_multi_stem_ensemble:
            self.primary_stem, self.secondary_stem = main_model_primary, secondary_stem(main_model_primary)

        if self.is_gpu_conversion >= 0:
            if mps_available:
                self.device, self.is_other_gpu = 'mps', True
            else:
                device_prefix = None
                if self.device_set != DEFAULT:
                    device_prefix = CUDA_DEVICE#DIRECTML_DEVICE if self.is_use_opencl and directml_available else CUDA_DEVICE

                # if directml_available and self.is_use_opencl:
                #     self.device = torch_directml.device() if not device_prefix else f'{device_prefix}:{self.device_set}'
                #     self.is_other_gpu = True
                if cuda_available:# and not self.is_use_opencl:
                    self.device = CUDA_DEVICE if not device_prefix else f'{device_prefix}:{self.device_set}'
                    self.run_type = ['CUDAExecutionProvider']

        if model_data.process_method == MDX_ARCH_TYPE:
            self.is_mdx_ckpt = model_data.is_mdx_ckpt
            self.primary_model_name, self.primary_sources = self.cached_source_callback(MDX_ARCH_TYPE, model_name=self.model_basename)
            self.is_denoise = model_data.is_denoise#
            self.is_denoise_model = model_data.is_denoise_model#
            self.is_mdx_c_seg_def = model_data.is_mdx_c_seg_def#
            self.mdx_batch_size = model_data.mdx_batch_size
            self.compensate = model_data.compensate
            self.mdx_segment_size = model_data.mdx_segment_size
            
            if self.is_mdx_c:
                if not self.is_4_stem_ensemble:
                    self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
                    self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem
            else:
                self.dim_f, self.dim_t = model_data.mdx_dim_f_set, 2**model_data.mdx_dim_t_set
                
            self.check_label_secondary_stem_runs()
            self.n_fft = model_data.mdx_n_fft_scale_set
            self.chunks = model_data.chunks
            self.margin = model_data.margin
            self.adjust = 1
            self.dim_c = 4
            self.hop = 1024

        if model_data.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_stems = model_data.demucs_stems if not main_process_method in [MDX_ARCH_TYPE, VR_ARCH_TYPE] else None
            self.secondary_model_4_stem = model_data.secondary_model_4_stem
            self.secondary_model_4_stem_scale = model_data.secondary_model_4_stem_scale
            self.is_chunk_demucs = model_data.is_chunk_demucs
            self.segment = model_data.segment
            self.demucs_version = model_data.demucs_version
            self.demucs_source_list = model_data.demucs_source_list
            self.demucs_source_map = model_data.demucs_source_map
            self.is_demucs_combine_stems = model_data.is_demucs_combine_stems
            self.demucs_stem_count = model_data.demucs_stem_count
            self.pre_proc_model = model_data.pre_proc_model
            self.device = "cpu" if self.is_other_gpu and not self.demucs_version in [DEMUCS_V3, DEMUCS_V4] else self.device

            self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
            self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem

            if (self.is_multi_stem_ensemble or self.is_4_stem_ensemble) and not self.is_secondary_model:
                self.is_return_dual = False
            
            if self.is_multi_stem_ensemble and main_model_primary:
                self.is_4_stem_ensemble = False
                if main_model_primary in self.demucs_source_map.keys():
                    self.primary_stem = main_model_primary
                    self.secondary_stem = secondary_stem(main_model_primary)
                elif secondary_stem(main_model_primary) in self.demucs_source_map.keys():
                    self.primary_stem = secondary_stem(main_model_primary)
                    self.secondary_stem = main_model_primary

            if self.is_secondary_model and not process_data['is_ensemble_master']:
                if not self.demucs_stem_count == 2 and model_data.primary_model_primary_stem == INST_STEM:
                    self.primary_stem = VOCAL_STEM
                    self.secondary_stem = INST_STEM
                else:
                    self.primary_stem = model_data.primary_model_primary_stem
                    self.secondary_stem = secondary_stem(self.primary_stem)

            self.shifts = model_data.shifts
            self.is_split_mode = model_data.is_split_mode if not self.demucs_version == DEMUCS_V4 else True
            self.primary_model_name, self.primary_sources = self.cached_source_callback(DEMUCS_ARCH_TYPE, model_name=self.model_basename)

        if model_data.process_method == VR_ARCH_TYPE:
            self.check_label_secondary_stem_runs()
            self.primary_model_name, self.primary_sources = self.cached_source_callback(VR_ARCH_TYPE, model_name=self.model_basename)
            # breakpoint()
            self.mp = model_data.vr_model_param
            self.high_end_process = model_data.is_high_end_process
            self.is_tta = model_data.is_tta
            self.is_post_process = model_data.is_post_process
            self.is_gpu_conversion = model_data.is_gpu_conversion
            self.batch_size = model_data.batch_size
            self.window_size = model_data.window_size
            self.input_high_end_h = None
            self.input_high_end = None
            self.post_process_threshold = model_data.post_process_threshold
            self.aggressiveness = {'value': model_data.aggression_setting, 
                                   'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                   'aggr_correction': self.mp.param.get('aggr_correction')}
            
    def check_label_secondary_stem_runs(self):

        # For ensemble master that's not a 4-stem ensemble, and not mdx_c
        if self.process_data['is_ensemble_master'] and not self.is_4_stem_ensemble and not self.is_mdx_c:
            if self.ensemble_primary_stem != self.primary_stem:
                self.is_primary_stem_only, self.is_secondary_stem_only = self.is_secondary_stem_only, self.is_primary_stem_only
            
        # For secondary models
        if self.is_pre_proc_model or self.is_secondary_model:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
            

    def running_inference_progress_bar(self, length, is_match_mix=False):
        if not is_match_mix:
            self.progress_value += 1

            if (0.8/length*self.progress_value) >= 0.8:
                length = self.progress_value + 1
  
            self.set_progress_bar(0.1, (0.8/length*self.progress_value))
        

    def process_vocal_split_chain(self, sources: dict):
        
        def is_valid_vocal_split_condition(master_vocal_source):
            """Checks if conditions for vocal split processing are met."""
            conditions = [
                isinstance(master_vocal_source, np.ndarray),
                self.vocal_split_model,
                not self.is_ensemble_mode,
                not self.is_karaoke,
                not self.is_bv_model
            ]
            return all(conditions)
        
        # Retrieve sources from the dictionary with default fallbacks
        master_inst_source = sources.get(INST_STEM, None)
        master_vocal_source = sources.get(VOCAL_STEM, None)

        # Process the vocal split chain if conditions are met
        if is_valid_vocal_split_condition(master_vocal_source):
            process_chain_model(
                self.vocal_split_model,
                self.process_data,
                vocal_stem_path=self.master_vocal_path,
                master_vocal_source=master_vocal_source,
                master_inst_source=master_inst_source
            )
  
    def process_secondary_stem(self, stem_source, secondary_model_source=None, model_scale=None):
        if not self.is_secondary_model:
            if self.is_secondary_model_activated and isinstance(secondary_model_source, np.ndarray):
                secondary_model_scale = model_scale if model_scale else self.secondary_model_scale
                stem_source = spec_utils.average_dual_sources(stem_source, secondary_model_source, secondary_model_scale)
  
        return stem_source
    
    def final_process(self, stem_path, source, secondary_source, stem_name, samplerate):
        source = self.process_secondary_stem(source, secondary_source)
        self.write_audio(stem_path, source, samplerate, stem_name=stem_name)
        
        return {stem_name: source}
    
    def write_audio(self, stem_path: str, stem_source, samplerate, stem_name=None):

        def save_audio_file(path, source):
            source = spec_utils.normalize(source, self.is_normalization)
            sf.write(path, source, samplerate, subtype=self.wav_type_set)

            if is_not_ensemble:
                save_format(path, self.save_format, self.mp3_bit_set)

        def save_voc_split_instrumental(stem_name, stem_source, is_inst_invert=False):
            inst_stem_name = "Instrumental (With Lead Vocals)" if stem_name == LEAD_VOCAL_STEM else "Instrumental (With Backing Vocals)"
            inst_stem_path_name = LEAD_VOCAL_STEM_I if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_I
            inst_stem_path = self.audio_file_base_voc_split(INST_STEM, inst_stem_path_name)
            stem_source = -stem_source if is_inst_invert else stem_source
            inst_stem_source = spec_utils.combine_arrarys([self.master_inst_source, stem_source], is_swap=True)
            save_with_message(inst_stem_path, inst_stem_name, inst_stem_source)

        def save_voc_split_vocal(stem_name, stem_source):
            voc_split_stem_name = LEAD_VOCAL_STEM_LABEL if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_LABEL
            voc_split_stem_path = self.audio_file_base_voc_split(VOCAL_STEM, stem_name)
            save_with_message(voc_split_stem_path, voc_split_stem_name, stem_source)

        def save_with_message(stem_path, stem_name, stem_source):
            is_deverb = self.is_deverb_vocals and (
                self.deverb_vocal_opt == stem_name or
                (self.deverb_vocal_opt == 'ALL' and 
                (stem_name == VOCAL_STEM or stem_name == LEAD_VOCAL_STEM_LABEL or stem_name == BV_VOCAL_STEM_LABEL)))

            
            if is_deverb and is_not_ensemble:
                deverb_vocals(stem_path, stem_source)
            
            save_audio_file(stem_path, stem_source)
            
        def deverb_vocals(stem_path:str, stem_source):
            stem_source_deverbed, stem_source_2 = vr_denoiser(stem_source, self.device, is_deverber=True, model_path=self.DEVERBER_MODEL)
            save_audio_file(stem_path.replace(".wav", "_deverbed.wav"), stem_source_deverbed)
            save_audio_file(stem_path.replace(".wav", "_reverb_only.wav"), stem_source_2)
            
        is_bv_model_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == LEAD_VOCAL_STEM)
        is_bv_rebalance_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == BV_VOCAL_STEM)
        is_no_vocal_save = self.is_inst_only_voc_splitter and (stem_name == VOCAL_STEM or stem_name == BV_VOCAL_STEM or stem_name == LEAD_VOCAL_STEM) or is_bv_model_lead
        is_not_ensemble = (not self.is_ensemble_mode or self.is_vocal_split_model)
        is_do_not_save_inst = (self.is_save_vocal_only and self.is_sec_bv_rebalance and stem_name == INST_STEM)

        if is_bv_rebalance_lead:
            master_voc_source = spec_utils.match_array_shapes(self.master_vocal_source, stem_source, is_swap=True)
            bv_rebalance_lead_source = stem_source-master_voc_source


        if not is_bv_model_lead and not is_do_not_save_inst:
            if self.is_vocal_split_model or not self.is_secondary_model:
                if self.is_vocal_split_model and not self.is_inst_only_voc_splitter:
                    save_voc_split_vocal(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_vocal(LEAD_VOCAL_STEM, bv_rebalance_lead_source)
                else:
                    if not is_no_vocal_save:
                        save_with_message(stem_path, stem_name, stem_source)
                    
                if self.is_save_inst_vocal_splitter and not self.is_save_vocal_only:
                    save_voc_split_instrumental(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_instrumental(LEAD_VOCAL_STEM, bv_rebalance_lead_source, is_inst_invert=True)

                self.set_progress_bar(0.95)

        if stem_name == VOCAL_STEM:
            self.master_vocal_path = stem_path

    def pitch_fix(self, source, sr_pitched, org_mix):
        semitone_shift = self.semitone_shift
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
        source = spec_utils.match_array_shapes(source, org_mix)
        return source
    
    def match_frequency_pitch(self, mix):
        source = mix
        if self.is_match_frequency_pitch and self.is_pitch_change:
            source, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
            source = self.pitch_fix(source, sr_pitched, mix)

        return source

class SeperateMDX(SeperateAttributes):        

    def seperate(self):
        samplerate = 44100
    
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, source = self.primary_sources
            self.load_cached_sources()
        else:

            if self.is_mdx_ckpt:
                model_params = torch.load(self.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
                self.dim_c, self.hop = model_params['dim_c'], model_params['hop_length']
                separator = MdxnetSet.ConvTDFNet(**model_params)
                self.model_run = separator.load_from_checkpoint(self.model_path).to(self.device).eval()
            else:
                if self.mdx_segment_size == self.dim_t and not self.is_other_gpu:
                    ort_ = ort.InferenceSession(self.model_path, providers=self.run_type)
                    self.model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
                else:
                    self.model_run = ConvertModel(load(self.model_path))
                    self.model_run.to(self.device).eval()

            mix = prepare_mix(self.audio_file)
            
            source = self.demix(mix)
        

        mdx_net_cut = True if self.primary_stem in MDX_NET_FREQ_CUT and self.is_match_frequency_pitch else False

        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)
        
        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                raw_mix = self.demix(self.match_frequency_pitch(mix), is_match_mix=True) if mdx_net_cut else self.match_frequency_pitch(mix)
                self.secondary_source = spec_utils.invert_stem(raw_mix, source) if self.is_invert_spec else mix.T-source.T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)
        
        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')

            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)
        
        clear_gpu_cache()

        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)

        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources

    def initialize_model_settings(self):
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.mdx_segment_size-1)
        self.gen_size = self.chunk_size-2*self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)

    def demix(self, mix, is_match_mix=False):
        self.initialize_model_settings()
        
        org_mix = mix
        tar_waves_ = []

        if is_match_mix:
            chunk_size = self.hop * (256-1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap_mdx
            
            if self.is_pitch_change:
                mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

        gen_size = chunk_size-2*self.trim

        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

        step = self.chunk_size - self.n_fft if overlap == DEFAULT else int((1 - overlap) * chunk_size)
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step

        for i in range(0, mixture.shape[-1], step):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])

            chunk_size_actual = end - start

            if overlap == 0:
                window = None
            else:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.device)
            mix_waves = mix_part.split(self.mdx_batch_size)
            
            with torch.no_grad():
                for mix_wave in mix_waves:
                    self.running_inference_progress_bar(total_chunks, is_match_mix=is_match_mix)

                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)
                    
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window 
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., :end-start]
            
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim:-self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
        
        source = tar_waves[:,0:None]

        if self.is_pitch_change and not is_match_mix:
            source = self.pitch_fix(source, sr_pitched, org_mix)

        source = source if is_match_mix else source*self.compensate
        if self.is_denoise_model and not is_match_mix:
            if NO_STEM in self.primary_stem_native or self.primary_stem_native == INST_STEM:
                if org_mix.shape[1] != source.shape[1]:
                    source = spec_utils.match_array_shapes(source, org_mix)
                source = org_mix - vr_denoiser(org_mix-source, self.device, model_path=self.DENOISER_MODEL)
            else:
                source = vr_denoiser(source, self.device, model_path=self.DENOISER_MODEL)

        return source

    def run_model(self, mix, is_match_mix=False):
        
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)

        return self.stft.inverse(torch.tensor(spec_pred).to(self.device)).cpu().detach().numpy()

class SeperateMDXC(SeperateAttributes):        

    def seperate(self):
        samplerate = 44100
        sources = None
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, sources = self.primary_sources
        else:
            mix = prepare_mix(self.audio_file)
            sources = self.demix(mix)

        stem_list = [self.mdx_c_configs.training.target_instrument] if self.mdx_c_configs.training.target_instrument else [i for i in self.mdx_c_configs.training.instruments]

        if self.is_secondary_model:
            if self.is_pre_proc_model:
                self.mdxnet_stem_select = stem_list[0]
            else:
                self.mdxnet_stem_select = self.main_model_primary_stem_4_stem if self.main_model_primary_stem_4_stem else self.primary_model_primary_stem
            self.primary_stem = self.mdxnet_stem_select
            self.secondary_stem = secondary_stem(self.mdxnet_stem_select)
            self.is_primary_stem_only, self.is_secondary_stem_only = False, False

        is_all_stems = self.mdxnet_stem_select == ALL_STEMS
        is_not_ensemble_master = not self.process_data['is_ensemble_master']
        is_not_single_stem = not len(stem_list) <= 2
        is_not_secondary_model = not self.is_secondary_model
        is_ensemble_4_stem = self.is_4_stem_ensemble and is_not_single_stem
        if (is_all_stems and is_not_ensemble_master and is_not_single_stem and is_not_secondary_model) or is_ensemble_4_stem and not self.is_pre_proc_model:
            for stem in stem_list:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem}).wav')
                self.primary_source = sources[stem].T
                self.write_audio(primary_stem_path, self.primary_source, samplerate, stem_name=stem)
                
                if stem == VOCAL_STEM and not self.is_sec_bv_rebalance:
                    self.process_vocal_split_chain({VOCAL_STEM:stem})
        else:
            if len(stem_list) == 1:
                source_primary = sources  
            else:
                source_primary = sources[stem_list[0]] if self.is_multi_stem_ensemble and len(stem_list) == 2 else sources[self.mdxnet_stem_select]
            if self.is_secondary_model_activated and self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, 
                                                                                                         self.process_data, 
                                                                                                         main_process_method=self.process_method, 
                                                                                                         main_model_primary=self.primary_stem)
            if not self.is_primary_stem_only:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
                if not isinstance(self.secondary_source, np.ndarray):
                    
                    if self.is_mdx_combine_stems and len(stem_list) >= 2:
                        if len(stem_list) == 2:
                            secondary_source = sources[self.secondary_stem]
                        else:
                            sources.pop(self.primary_stem)
                            next_stem = next(iter(sources))
                            secondary_source = np.zeros_like(sources[next_stem])
                            for v in sources.values():
                                secondary_source += v
                                
                        self.secondary_source = secondary_source.T 
                    else:
                        self.secondary_source, raw_mix = source_primary, self.match_frequency_pitch(mix)
                        self.secondary_source = spec_utils.to_shape(self.secondary_source, raw_mix.shape)
                    
                        if self.is_invert_spec:
                            self.secondary_source = spec_utils.invert_stem(raw_mix, self.secondary_source)
                        else:
                            self.secondary_source = (-self.secondary_source.T+raw_mix.T)
                            
                self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)    


            if not self.is_secondary_stem_only:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = source_primary.T

                self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)

        clear_gpu_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        self.process_vocal_split_chain(secondary_sources)
        
        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources

    def demix(self, mix):
        sr_pitched = 441000
        org_mix = mix
        if self.is_pitch_change:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

        model = TFC_TDF_net(self.mdx_c_configs, device=self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        model.to(self.device).eval()
        mix = torch.tensor(mix, dtype=torch.float32)

        try:
            S = model.num_target_instruments
        except Exception as e:
            S = model.module.num_target_instruments

        mdx_segment_size = self.mdx_c_configs.inference.dim_t if self.is_mdx_c_seg_def else self.mdx_segment_size
        
        batch_size = self.mdx_batch_size
        chunk_size = self.mdx_c_configs.audio.hop_length * (mdx_segment_size - 1)
        overlap = self.overlap_mdx23

        hop_size = chunk_size // overlap
        mix_shape = mix.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size
        mix = torch.cat([torch.zeros(2, chunk_size - hop_size), mix, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)

        chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)
        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        X = torch.zeros(S, *mix.shape) if S > 1 else torch.zeros_like(mix)
        X = X.to(self.device)

        with torch.no_grad():
            cnt = 0
            for batch in batches:
                self.running_inference_progress_bar(len(batches))
                x = model(batch.to(self.device))
                
                for w in x:
                    X[..., cnt * hop_size : cnt * hop_size + chunk_size] += w
                    cnt += 1

        estimated_sources = X[..., chunk_size - hop_size:-(pad_size + chunk_size - hop_size)] / overlap
        del X
        pitch_fix = lambda s:self.pitch_fix(s, sr_pitched, org_mix)

        if S > 1:
            sources = {k: pitch_fix(v) if self.is_pitch_change else v for k, v in zip(self.mdx_c_configs.training.instruments, estimated_sources.cpu().detach().numpy())}
            del estimated_sources
            if self.is_denoise_model:
                if VOCAL_STEM in sources.keys() and INST_STEM in sources.keys():
                    sources[VOCAL_STEM] = vr_denoiser(sources[VOCAL_STEM], self.device, model_path=self.DENOISER_MODEL)
                    if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                        sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
                    sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
                            
            return sources
        else:
            est_s = estimated_sources.cpu().detach().numpy()
            del estimated_sources
            return pitch_fix(est_s) if self.is_pitch_change else est_s

class SeperateVR(SeperateAttributes):        

    def seperate(self):
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            y_spec, v_spec = self.primary_sources
            self.load_cached_sources()
        else:

            device = self.device

            nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))
            # breakpoint()
            if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
                self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, 
                                                      nn_arch_size, 
                                                      nout=self.model_capacity[0], 
                                                      nout_lstm=self.model_capacity[1])
                self.is_vr_51_model = True
            else:
                self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
                            
            self.model_run.load_state_dict(torch.load(self.model_path, map_location=DEVICE)) 
            self.model_run.to(device) 

            y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
            # breakpoint()
        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)
        
        # breakpoint()
        if not self.is_secondary_stem_only:
            # breakpoint()
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T  # , res_type="kaiser_best"
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, 44100)  

        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T  # , res_type="kaiser_best"
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, 44100)
            
        clear_gpu_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)
        
        if self.is_secondary_model:
            return secondary_sources
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(self.audio_file, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            if OPERATING_SYSTEM == 'Darwin':
                wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            else:
                wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution) # "kaiser_best"
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=self.is_vr_51_model)
        
        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring') and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)       
        else:
            wav = cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=self.is_vr_51_model)
            
        return wav


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    bands_n = len(mp.param['band'])    
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param['band'][d]
        spec_s = np.ndarray(shape=(2, bp['n_fft'] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp['crop_stop'] - bp['crop_start']
        spec_s[:, bp['crop_start']:bp['crop_stop'], :] = spec_m[:, offset:offset+h, :]
                
        offset += h
        if d == bands_n: # higher
            if extra_bins_h: # if --high_end_process bypass
                max_bin = bp['n_fft'] // 2
                spec_s[:, max_bin-extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp['hpf_start'] > 0:
                if is_v51_model:
                    spec_s *= spec_utils.get_hp_filter_mask(spec_s.shape[1], bp['hpf_start'], bp['hpf_stop'] - 1)
                else:
                    spec_s = spec_utils.fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
            if bands_n == 1:
                wave = spec_utils.spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model)
            else:
                wave = np.add(wave, spec_utils.spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model))
        else:
            sr = mp.param['band'][d+1]['sr']
            if d == 1: # lower
                if is_v51_model:
                    spec_s *= spec_utils.get_lp_filter_mask(spec_s.shape[1], bp['lpf_start'], bp['lpf_stop'])
                else:
                    spec_s = spec_utils.fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave = librosa.resample(spec_utils.spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model), bp['sr'], sr, res_type="kaiser_best") #   "sinc_fastest"
            else: # mid
                if is_v51_model:
                    spec_s *= spec_utils.get_hp_filter_mask(spec_s.shape[1], bp['hpf_start'], bp['hpf_stop'] - 1)
                    spec_s *= spec_utils.get_lp_filter_mask(spec_s.shape[1], bp['lpf_start'], bp['lpf_stop'])
                else:
                    spec_s = spec_utils.fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
                    spec_s = spec_utils.fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                    
                wave2 = np.add(wave, spec_utils.spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model))
                wave = librosa.resample(wave2, bp['sr'], sr, res_type="kaiser_best") #   "sinc_fastest"
        
    return wave


def process_secondary_model(secondary_model: ModelData, 
                            process_data, 
                            main_model_primary_stem_4_stem=None, 
                            is_source_load=False, 
                            main_process_method=None, 
                            is_pre_proc_model=False, 
                            is_return_dual=True, 
                            main_model_primary=None):
        
    if not is_pre_proc_model:
        process_iteration = process_data['process_iteration']
        process_iteration()
    
    if secondary_model.process_method == VR_ARCH_TYPE:
        seperator = SeperateVR(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, main_model_primary=main_model_primary)
    if secondary_model.process_method == MDX_ARCH_TYPE:
        if secondary_model.is_mdx_c:
            seperator = SeperateMDXC(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, is_return_dual=is_return_dual, main_model_primary=main_model_primary)
        else:
            seperator = SeperateMDX(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, main_model_primary=main_model_primary)
    if secondary_model.process_method == DEMUCS_ARCH_TYPE:
        seperator = SeperateDemucs(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, is_return_dual=is_return_dual, main_model_primary=main_model_primary)
        
    secondary_sources = seperator.seperate()

    if type(secondary_sources) is dict and not is_source_load and not is_pre_proc_model:
        return gather_sources(secondary_model.primary_model_primary_stem, secondary_stem(secondary_model.primary_model_primary_stem), secondary_sources)
    else:
        return secondary_sources
    


class AudioProcessor:
    def __init__(self):
        self.device = DEVICE
        
    def process_audio_files(self, input_files, output_base_dir):
        """
        Process a list of audio files through the complete pipeline:
        1. MDX23C-8KFFT-InstVoc_HQ.ckpt for initial separation
        2. 5_HP-Karaoke-UVR.pth for vocal processing
        3. UVR-De-Echo-Normal.pth for echo removal
        
        Args:
            input_files (list): List of paths to input audio files
            output_base_dir (str): Base directory for outputs
        """
        # Create output directories
        instrumental_dir = os.path.join(output_base_dir, "instrumentals")
        intermediate_vocals_dir = os.path.join(output_base_dir, "intermediate_vocals")
        intermediate_hp_vocals_dir = os.path.join(output_base_dir, "hp_vocals")
        final_vocals_dir = os.path.join(output_base_dir, "final_vocals")
        
        os.makedirs(instrumental_dir, exist_ok=True)
        os.makedirs(intermediate_vocals_dir, exist_ok=True)
        os.makedirs(intermediate_hp_vocals_dir, exist_ok=True)
        os.makedirs(final_vocals_dir, exist_ok=True)
        
        for input_file in input_files:
            # try:
                print(f"\nProcessing {input_file}...")
                base_name = Path(input_file).stem
                
                # Step 1: Initial vocal/instrumental separation with MDX model
                start_time = time.time()
                print("\nStep 1: Initial separation using MDX23C-8KFFT-InstVoc_HQ.ckpt...")
                mdx_model = ModelData(
                    model_name="MDX23C-8KFFT-InstVoc_HQ.ckpt",
                    selected_process_method=MDX_ARCH_TYPE
                )
                mdx_model.mdx_segment_size = 256
                mdx_model.overlap_mdx23 = 8
                # mdx_model.is_mdx_c = False

                # ——— monkey‐patch the missing dims ———
                # freq bins comes from your MODEL_PARAMS: 2048
                mdx_model.mdx_dim_f_set      = mdx_model.model_data.get("num_freq_bins", 2048)
                # make dim_t so that 2**dim_t == segment_size (i.e. 2**8 == 256)
                mdx_model.mdx_dim_t_set      = int(math.log2(mdx_model.mdx_segment_size))
                # FFT size from your MODEL_PARAMS: 8192
                mdx_model.mdx_n_fft_scale_set = mdx_model.model_data.get("fft_size", 8192)
                process_data = {
                    'audio_file': str(input_file),
                    'audio_file_base': base_name,
                    'export_path': instrumental_dir, # Save instrumental here
                    'set_progress_bar': lambda x, y=None: None,
                    'process_iteration': None,
                    'cached_source_callback': lambda *args, **kwargs: (None, None),
                    'is_ensemble_master': False,
                    'list_all_models': [],
                    'is_4_stem_ensemble': False
                }
                # Run separation, save instrumental and get vocal path
                separator = SeperateMDXC(mdx_model, process_data)
                separator.seperate()
                vocal_path = os.path.join(separator.export_path, f'{separator.audio_file_base}_({separator.primary_stem}).wav')

                # Move vocal to intermediate directory
                mdx_vocal_path = os.path.join(intermediate_vocals_dir, f"{base_name}_(Vocals).wav")
                os.rename(vocal_path, mdx_vocal_path)

                end_time1 = time.time()
                print(f"Initial separation complete. Time used: {end_time1 - start_time}")

                # Step 2: Process vocal with HP-Karaoke model
                print("\nStep 2: Processing vocals with 5_HP-Karaoke-UVR model ...")
                root.is_secondary_stem_only_var = DummyVar(True)
                hp_model = ModelData(
                    model_name="5_HP-Karaoke-UVR",
                    selected_process_method=VR_ARCH_TYPE,
                )
                # hp_model.window_size = 512
                # hp_model.aggression_setting = 0.75
                hp_model.window_size = 320
                hp_model.aggression_setting = 5
                hp_model.is_tta = True
                hp_model.is_post_process = True
                hp_model.post_process_threshold = 0.2
                hp_model.batch_size = 64
                
                mdx_vocal_path = f"{output_base_dir}/intermediate_vocals/{base_name}_(Vocals).wav"
                process_data['audio_file'] = mdx_vocal_path
                process_data['export_path'] = intermediate_hp_vocals_dir
                
                separator = SeperateVR(hp_model, process_data)
                separator.seperate()
                hp_vocal_path = os.path.join(separator.export_path, f'{separator.audio_file_base}_({separator.secondary_stem}).wav')
                end_time2 = time.time()
                print(f"HP-Karaoke processing complete. {end_time2 - end_time1}")
                
                # Step 3: Remove echo
                print("\nStep 3: Removing echo with UVR-De-Echo-Normal model ...")
                root.is_secondary_stem_only_var = DummyVar(False)
                # root.is_primary_stem_only_var = DummyVar(True)
                # root.is_deverb_vocals_var = DummyVar(True)
                # root.is_primary_stem_only_var   = DummyVar(True)   # 不要只存 primary
                # root.is_secondary_stem_only_var = DummyVar(True)    # 只存 secondary (dry vocal)
                deecho_model = ModelData(
                    model_name="UVR-De-Echo-Normal",
                    # model_name="UVR-DeEcho-DeReverb",
                    # model_name="UVR-De-Echo-Aggressive",
                    selected_process_method=VR_ARCH_TYPE,
                )
                # # deecho_model.window_size = 512
                # # deecho_model.aggression_setting = 0.5
                # deecho_model.window_size = 320
                # deecho_model.aggression_setting = 5
                # deecho_model.is_tta = True
                # deecho_model.is_post_process = True
                # deecho_model.post_process_threshold = 0.1
                # deecho_model.batch_size = 4

                # deecho_model.window_size        = 512
                # deecho_model.aggression_setting = 0.5
                # deecho_model.is_tta             = False
                # deecho_model.is_post_process    = False
                # deecho_model.post_process_threshold = 0.1
                # deecho_model.batch_size         = 4

                # deecho_model.window_size        = 320
                # deecho_model.aggression_setting = 5
                # deecho_model.is_tta             = True
                # deecho_model.is_post_process    = True
                # deecho_model.post_process_threshold = 0.1
                # deecho_model.batch_size         = 64

                # deecho_model.window_size        = 512
                # deecho_model.aggression_setting = 5    
                # deecho_model.is_tta             = False
                # deecho_model.is_post_process    = False
                # deecho_model.batch_size         = 64

                process_data['audio_file'] = hp_vocal_path
                process_data['export_path'] = final_vocals_dir
                
                separator = SeperateVR(deecho_model, process_data)
                separator.seperate()

                end_time3 = time.time()
                print(f"Echo removal complete. {end_time3 - end_time2}")

                print(f"\nCompleted processing {input_file}. Total time: {end_time3 - start_time}")
                
            # except Exception as e:
            #     print(f"Error processing {input_file}: {e}")
            #     continue

def main():
    parser = argparse.ArgumentParser(description='Process audio files (wav, mp3, m4a, flac) for vocal separation and enhancement.')
    parser.add_argument('input_dir', help='Directory containing input audio files (wav, mp3, m4a, flac).')
    parser.add_argument('output_dir', help='Directory for output files.')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    # Get list of input files
    supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    input_files = []
    for ext in supported_extensions:
        input_files.extend(Path(input_dir).glob(ext))

    if not input_files:
        print(f"No supported audio files found in {input_dir}. Supported formats: wav, mp3, m4a, flac.")
        return

    print(f"\nFound {len(input_files)} audio files to process.")
    
    processor = AudioProcessor()
    processor.process_audio_files(input_files, output_dir)

if __name__ == "__main__":
    main() 