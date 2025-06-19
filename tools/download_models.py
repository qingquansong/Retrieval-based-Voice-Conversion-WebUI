import os
from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

BASE_DIR = Path(__file__).resolve().parent.parent


def dl_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)
        with open(dir_name / model_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    print("Downloading hubert_base.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/hubert")
    print("Downloading rmvpe.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rmvpe")
    # print("Downloading vocals.onnx...")
    # dl_model(
    #     RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
    #     "vocals.onnx",
    #     BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
    # )

    rvc_models_dir = BASE_DIR / "assets/pretrained"

    print("Downloading pretrained models:")

    model_names = [
        # "D32k.pth",
        "D40k.pth",
        # "D48k.pth",
        # "G32k.pth",
        # "G40k.pth",
        "G48k.pth",
        # "f0D32k.pth",
        # "f0D40k.pth",
        "f0D48k.pth",
        # "f0G32k.pth",
        # "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir)

    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"

    print("Downloading pretrained models v2:")

    for model in model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, rvc_models_dir)

    print("Downloading uvr5_weights:")

    # rvc_models_dir = BASE_DIR / "assets/uvr5_weights"
    uvr5_mdx_model_weights_dir = BASE_DIR /  "ultimatevocalremovergui/models/MDX_Net_Models"
    model_names = [
        "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model("https://huggingface.co/QQSong/UVR5/resolve/main/", model, uvr5_mdx_model_weights_dir)

    uvr5_vr_model_weights_dir = BASE_DIR /  "ultimatevocalremovergui/models/VR_Models"
    model_names = [
        "5_HP-Karaoke-UVR.pth",
        "6_HP-Karaoke-UVR.pth",
        "UVR-De-Echo-Aggressive.pth",
        "UVR-De-Echo-Normal.pth",
        "UVR-DeEcho-DeReverb.pth",
        "UVR-DeNoise.pth",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model("https://huggingface.co/QQSong/UVR5/resolve/main/", model, uvr5_vr_model_weights_dir)

    print("All models downloaded!")
