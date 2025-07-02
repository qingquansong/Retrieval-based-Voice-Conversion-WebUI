import argparse
import os
import json


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-singer",
        "--singer_name",
        type=str,
        required=True,
        help="The singer name" # this is used to generate the experimentation dir to save the training results.
    )
    parser.add_argument(
        "-sr", 
        "--sampling_rate", type=int, required=True, help="Sampling rate"
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
        "--the speaker_id",
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
        help="How to allocate the gpus when runing rmvpe. e.g. 0-0-1, gpu 0 has 2 processes, and gpu 1 has 1 process. Default to 2 process on each GPU"  #generate the default value by "%s-%s" % (gpus, gpus),
    )

    args = parser.parse_args()
    singer_name = args.singer_name
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
    hparams.sr2 = args.sampling_rate
    # exp_dir1,
    # sr2,
    # if_f0_3,
    # trainset_dir4,
    # spk_id5,
    # np7,
    # f0method8,
    # save_epoch10,
    # total_epoch11,
    # batch_size12,
    # if_save_latest13,
    # pretrained_G14,
    # pretrained_D15,
    # gpus16,
    # if_cache_gpu17,
    # if_save_every_weights18,
    # version19,
    # gpus_rmvpe,
    hparams.name = single_name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = "%s/filelist.txt" % experiment_dir
    return hparams


class HParams:

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
