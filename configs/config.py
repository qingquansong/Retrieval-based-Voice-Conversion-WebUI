import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count

import torch
from train_pipeline.utils import HParams

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import logging

logger = logging.getLogger(__name__)


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
            self.hparams
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )

        #define API arguments
        parser.add_argument(
        "-singer",
        "--singer_name",
        type=str,
        required=True,
        help="The singer name" # this is used to generate the experimentation dir to save the training results.
        )
        parser.add_argument(
            "-sr", 
            "--sampling_rate", 
            type=str,
            default='48k',
            help="Sampling rate"
        )
        parser.add_argument(
            "-f0",
            "--if_f0_guidance",
            type=int,
            default=1,
            help="Use f0 guidance(required for singing)" 
        )
        parser.add_argument(
            "-tdd",
            "--training_data_dir",
            type=str,
            required=True,
            help="The directory of the training data"
        )
        parser.add_argument(
            "-spk_id",
            "--speaker_id",
            type=int,
            required=True,
            help="The speaker id"
        )
        parser.add_argument(
            "-cur",
            "--cpu_usage_rate",
            type=float,
            default=0.6,
            help="The cpu usage rate" #This is used to generate how many cpu processes to be used when extract f0
        )
        parser.add_argument(
            "-f0m",
            "--f0_model",
            type=str,
            default="rmvpe_gpu",
            help="The f0 model to be used" #available models:  "pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"
        )
        parser.add_argument(
            "-scf",
            "--saving_checkpoint_frequency",
            type=int,
            default=20,
            help="The frequency to save the checkpoint file" 
        )
        parser.add_argument(
            "-te",
            "--total_epoch",
            type=int,
            default=100,
            help="The total number of epoch" 
        )
        parser.add_argument(
            "-bs",
            "--batch_size",
            type=int,
            default=20,
            help="The batch size in each epoch" 
        )
        parser.add_argument(
            "-slco",
            "--save_latest_checkpoint_only",
            type=int,
            default=1,
            help="Whether ONLY save the lastest checkpoint" #to save the space
        )
        parser.add_argument(
            "-bmgp",
            "--base_model_G_path",
            type=str,
            default="assets/pretrained_v2/f0G40k.pth",
            help="The base model G path, the model should be downloaded to this path." 
        )
        parser.add_argument(
            "-bmdp",
            "--base_model_D_path",
            type=str,
            default="assets/pretrained_v2/f0D40k.pth",
            help="The base model D path, the model should be downloaded to this path." 
        )
        parser.add_argument(
            "-gpus",
            "--gpus_to_use",
            type=str,
            required=True,
            help="Specify the gpus to use, if not given, use all. (e.g. 0-1-2, uses gpu 0, 1 and 2)" 
        )
        parser.add_argument(
            "--cache_training_data_to_gpu",
            type=int,
            default=0,
            help="Cache the training data in GPU VRAM for faster training. It would probably ONLY helps if training dataset is less than 10 minuites. Large training data size might not worth" 
        )
        parser.add_argument(
            "--save_inference_weights",
            type=int,
            default=1,
            help="Save the weights on every checkpoint to weights directory" 
        )
        parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v2",
            help="The base model version." 
        )
        parser.add_argument(
            "--rmvpe_gpus",
            type=str,
            required=True,
            help="How to allocate the gpus when runing rmvpe. e.g. 0-0-1, gpu 0 has 2 processes, and gpu 1 has 1 process. Default to 2 process on each GPU"  #generate the default value by "%s-%s" % (gpus, gpus),
        )

        cmd_opts = parser.parse_args()
        singer_name = cmd_opts.singer_name
        # 1. The training config initialized when infer_web.py is executed via config = Config()
        # 2. As part of step 1, the config in ./config/config.json is copied to ./config/inuse/config.json
        # 3. After step 2, it load the json file and save the configs to json dict in the config instance. Key is the config versiont/sampling_rate.json
        # 4. When training, the config with the specific version/sampling_rate.json is dumpped to exp_dir/config.json.
        # 5. When train.py is executed, the exp_dir/config.json is read into HParams() along with the input arguments passed into the command. 
        
        # For this particular parameter parser, I just need to parse the train.py parameters and pass it into train.py.
        # As long as this parser happens after the step 1 to 5 mentioned above, the training should work out of box.
        hparams = HParams()
        #train1key exp_dir1
        hparams.exp_dir1 = singer_name
        hparams.sr2 = cmd_opts.sampling_rate
        hparams.if_f0_3 = cmd_opts.if_f0_guidance
        hparams.trainset_dir4 = cmd_opts.training_data_dir
        hparams.spk_id5 = cmd_opts.speaker_id
        hparams.np7 = cmd_opts.cpu_usage_rate
        hparams.f0method8 = cmd_opts.f0_model
        hparams.save_epoch10 = cmd_opts.saving_checkpoint_frequency
        hparams.total_epoch11 = cmd_opts.total_epoch
        hparams.batch_size12 = cmd_opts.batch_size
        hparams.if_save_latest13 = cmd_opts.save_latest_checkpoint_only
        hparams.pretrained_G14 = cmd_opts.base_model_G_path
        hparams.pretrained_D15 = cmd_opts.base_model_G_path
        hparams.gpus16 = cmd_opts.gpus_to_use
        hparams.if_cache_gpu17 = cmd_opts.cache_training_data_to_gpu
        hparams.if_save_every_weights18 = cmd_opts.save_inference_weights
        hparams.version19 = cmd_opts.version
        hparams.gpus_rmvpe = cmd_opts.rmvpe_gpus

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
            hparams
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            with open(f"configs/inuse/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"configs/inuse/{config_file}", "w") as f:
                f.write(strr)
            logger.info("overwrite " + config_file)
        self.preprocess_per = 3.0
        logger.info("overwrite preprocess_per to %d" % (self.preprocess_per))

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            logger.info("Use DirectML instead")
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-dml",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
            # if self.device != "cpu":
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
        logger.info(
            "Half-precision floating-point: %s, device: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max
