# Extending the Prompting Modalities of STEVE-1-based Generative Agents

## Installation
I recomend using python 3.10 when running this code.
1. Follow the instructions to install the [STEVE-1 agent](https://github.com/Shalev-Lifshitz/STEVE-1)
2. Download this repository and place it in the same directory as the STEVE-1 installation: `git clone https://github.com/nlenzen/steve_multimodal`
   The directory structure should look like this:
   ```
    parent_dir
    ├── STEVE-1
    └── steve_multimodal

   ```
4. Change into the steve_multimodal directory: `cd minecraft_audio_video_clip`
5. Run `pip install requirements.txt`
6. All done!

## Generating gameplay video conditioned on audio
The audio prompts used in our experiments can be found in the directory `audio_prompts`.
1. To use you own audio prompts, place the them inside of this folder in the form of `.wav` files. For organization, the directory can have one layer of subdirectories in which the `.wav` files can be placed.
2. Download the [Audio-Video CLIP weights](https://drive.google.com/file/d/1Gj35wWq7mtvzcH27MH8T2Mxv0F5O0bpa/view?usp=drive_link), as well as the [audio-prior weights](https://drive.google.com/file/d/1fTQbOvUvisgJSZlvsN_KqIzoSy0j_HxC/view?usp=sharing) and place the insidem the folder `checkpoints`.
3. Start the generation by running `python minecraft_audio_video_clip/steve_interfaces/interface_run_agent.py` inside `parent_dir`. This will generate a gameplay video for each audio prompt in the `audio_prompts` directory.

4. To run an interactive session run `python minecraft_audio_video_clip/steve_interfaces/interface_interactive.py` inside `parent_dir`. Here you can specify a path to an audio prompt by clicking inside the video window and entering it into the console.
