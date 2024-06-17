import os
import sys

sys.path.append(os.getcwd())

import cv2
import torch
from tqdm import tqdm
import argparse

from minecraft_audio_video_clip.source.load import load_model

from interface_util import save_frames_as_video
from evaluation import TASKS, make_text_embeddings, add_task_prompts, Evaluator, make_agent, make_env, add_audio_prompts, eval_cosine_dist
from evaluation import MODEL_CFG, MODEL_PATH, DEVICE

FPS = 20    # 30
SEEDS = [0, 87400, 1184, 87630, 15405, 3254, 41745, 37, 654336, 8362362]


def run_agent(prompt_embed, gameplay_length, save_video_filepath, task_name, task, seed, env, agent):
    # Make sure seed is set if specified
    if seed is not None:
        env.seed(seed)
        print("Setting seed to {}".format(seed))
    obs = env.reset()

    # Setup
    gameplay_frames = []
    evaluator = Evaluator(obs, task_name, task)

    # Run agent in MineRL env
    for _ in tqdm(range(gameplay_length)):
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)
        frame = obs['pov']
        frame = cv2.resize(frame, (256, 160))
        gameplay_frames.append(frame)

        evaluator.update(obs)

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    evaluator.evaluate()

    return evaluator


def create_eval_video(prompt_embed, prompt_type, agent, env, seed, task_name, task, gameplay_length, save_dir):
    print(f'Generating {prompt_type} prompt video for task {task_name} with seed {seed}')
    video_save_path = os.path.join(save_dir, task_name, 'videos', prompt_type, f'seed-{seed}.mp4')
    if os.path.isfile(video_save_path):
        print('Video already exists - skipping')
    else:
        evaluator = run_agent(prompt_embed, gameplay_length, video_save_path, task_name, task, seed, env, agent)
        metric_savepath = os.path.join(save_dir, task_name, 'metrics', prompt_type, f'seed-{seed}.json')
        evaluator.save(metric_savepath)


def skip(task_name, prompt_types, save_dirpath):
    for seed in SEEDS:
        for prompt_type in prompt_types:
            path = os.path.join(save_dirpath, task_name, 'videos', prompt_type, f'seed-{seed}.mp4')
            if not os.path.isfile(path):
                return False
    return True


def eval_audio_text_visual(tasks, agent, gameplay_length, prompt_types, save_dirpath):
    for task_name in tasks.keys():
        print(f'Processing task {task_name}')

        # Check if videos need to be generated for the task before generating environment
        # Hopefully keeps my PC from crashing
        if skip(task_name, prompt_types, save_dirpath):
            print(f'Skipping task {task_name} - all videos already exist')
            continue

        task = tasks[task_name]
        audio_embed = task.get('audio_prompt')
        text_embed = task.get('text_prompt')
        visual_embed = task.get('visual_prompt')
        env = make_env(task)
        for seed in SEEDS:
            create_eval_video(audio_embed, 'audio', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            create_eval_video(text_embed, 'text', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            create_eval_video(visual_embed, 'visual', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            print('')
        del env


def eval_audios(tasks, agent, gameplay_length, prompt_types, save_dirpath):
    for task_name in tasks.keys():
        print(f'Processing task {task_name}')

        # Check if videos need to be generated for the task before generating environment
        # Hopefully keeps my PC from crashing
        if skip(task_name, prompt_types, save_dirpath):
            print(f'Skipping task {task_name} - all videos already exist')
            continue

        task = tasks[task_name]
        audio_embed_base = task.get('audio_prompt_base')
        audio_embed_one = task.get('audio_prompt_one')
        audio_embed_two = task.get('audio_prompt_two')
        env = make_env(task)
        for seed in SEEDS:
            create_eval_video(audio_embed_base, 'audio_base', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            create_eval_video(audio_embed_one, 'audio_one', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            create_eval_video(audio_embed_two, 'audio_two', agent, env, seed, task_name, task, gameplay_length, save_dirpath)
            print('')
        del env


def eval_dist(tasks, save_dirpath, prompt_types, eval_steve):
    model = load_model(MODEL_PATH, MODEL_CFG, DEVICE)
    for task_name in tasks.keys():
        if tasks[task_name]['key'] == 'dist':
            videos_path = os.path.join(save_dirpath, task_name, 'videos')
            for prompt_type in os.listdir(videos_path):
                if prompt_type in prompt_types:
                    path = os.path.join(videos_path, prompt_type)
                    for video in os.listdir(path):
                        print(f'Processing video {video} for prompt {prompt_type} for the task {task_name}')
                        video_path = os.path.join(path, video)
                        metric_path = os.path.join(save_dirpath, task_name, 'metrics', prompt_type, video)
                        metric_path = metric_path.replace('.mp4', '.json')
                        prompt = prompt_type + '_prompt' if eval_steve else prompt_type.replace('_', '_prompt_')
                        task_embedding = tasks[task_name][prompt]
                        eval_cosine_dist(task_embedding, video_path, metric_path, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='STEVE-1/data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='STEVE-1/data/weights/steve1/steve1.weights')
    parser.add_argument('--cond_scale', type=float, default=6.0)
    parser.add_argument('--gameplay_length', type=int, default=3000)
    parser.add_argument('--evaluate_steve', type=bool, default=False)
    parser.add_argument('--save_dirpath', type=str, default='videos/cvae_evaluation')
    parser.add_argument('--audio_prompt_path', type=str, default='datasets/embedded_prompts/audio_prompts_projection.npz')
    parser.add_argument('--visual_prompt_path', type=str, default='datasets/embedded_prompts/visual_embeddings.npz')
    parser.add_argument('--audio_prompt_base_path', type=str, default='datasets/embedded_prompts/audio_prompts_base_new.npz')
    parser.add_argument('--audio_prompt_1_path', type=str, default='datasets/embedded_prompts/audio_prompts_1.npz')
    parser.add_argument('--audio_prompt_2_path', type=str, default='datasets/embedded_prompts/audio_prompts_2.npz')

    args = parser.parse_args()
    agent = make_agent(args.in_model, args.in_weights, args.cond_scale)

    if args.evaluate_steve:
        prompt_types = ['audio', 'text', 'visual']
        tasks = make_text_embeddings(TASKS)
        tasks = add_task_prompts(tasks, args.audio_prompt_path, args.visual_prompt_path)
        eval_audio_text_visual(tasks, agent, args.gameplay_length, prompt_types, args.save_dirpath)
    else:
        prompt_types = ['audio_base', 'audio_one', 'audio_two']
        tasks = add_audio_prompts(TASKS, args.audio_prompt_base_path, args.audio_prompt_1_path, args.audio_prompt_2_path)
        eval_audios(tasks, agent, args.gameplay_length, prompt_types, args.save_dirpath)
    del agent
    eval_dist(tasks, args.save_dirpath, prompt_types, args.evaluate_steve)

