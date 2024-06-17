import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
from tqdm import tqdm

from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.VPT.agent import ENV_KWARGS
from steve1.mineclip_code.load_mineclip import load

from interface_util import make_agent
from config import MINECLIP_CONFIG, PRIOR_INFO

from load_util import save_json, load_json

from minecraft_audio_video_clip.source.dataloading.datasets import VideoDataset

# Constants ============================================================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/model/10_layers_extended_4.pth'
MODEL_CFG = 'minecraft_audio_video_clip/configs/10_layers.yaml'

FPS = 20
NUM_FRAMES = 16
BATCH_SIZE = 16
PREFETCH_FACTOR = 2
NUM_WORKERS = 2

# Tasks and their specifications =======================================================================================


TASKS = {
    'get_cobblestone': {
        'key': 'cobblestone',
        'name': 'Cobblestone',
        'pref_biome': 'plains',
        'inventory': [{'type': 'stone_pickaxe', 'quantity': 2}],
        'text_prompt': 'get cobblestone, mine stone, break stone',
        'task_collect': True
    },
    'get_dirt': {
        'key': 'dirt',
        'name': 'Dirt',
        'pref_biome': 'plains',
        'text_prompt': 'get dirt, mine dirt, obtain dirt',
        'task_collect': True
    },
    'go_swim': {
        'key': 'dist',
        'name': 'Distance',
        'pref_biome': 'plains',
        'text_prompt': 'go swimming',
        'task_collect': True
    },
    'go_underwater': {
        'key': 'dist',
        'name': 'Distance',
        'pref_biome': 'plains',
        'text_prompt': 'go underwater',
        'task_collect': True
    },
    'destroy_leaves': {
        'key': 'leaves',
        'name': "Leaves",
        'pref_biome': 'forest',
        'inventory': [{'type': 'shears', 'quantity': 32}],
        'text_prompt': 'break leaves, destroy leaves',
        'task_collect': True
    },
    'get_wood': {
        'key': '_log',
        'name': 'Wooden Logs',
        'pref_biome': 'forest',
        'text_prompt': 'chop a tree, get wood',
        'task_collect': True
    },
    'get_sand': {
        'key': 'sand',
        'name': 'Sand',
        'pref_biome': 'desert',
        'text_prompt': 'get sand, mine sand, obtain sand',
        'task_collect': True
    },
    'get_seeds': {
        'key': 'seed',
        'name': 'Seeds',
        'pref_biome': 'plains',
        'text_prompt': 'get seeds, obtain seeds, break grass',
        'task_collect': True
    }
}

# loading / generating prompts =========================================================================================


def make_text_embeddings(tasks):
    prior = load_vae_model(PRIOR_INFO)
    mineclip = load(MINECLIP_CONFIG, DEVICE)

    for task_name in tasks.keys():
        prompt = tasks[task_name]['text_prompt']
        prompt_embed = get_prior_embed(prompt, mineclip, prior, DEVICE)
        tasks[task_name]['text_prompt'] = prompt_embed

    return tasks


def add_task_prompts(tasks, audio, visual):
    audio_dict = np.load(audio)
    visual_dict = np.load(visual)

    for task_name in tasks.keys():
        tasks[task_name]['audio_prompt'] = audio_dict[task_name]
        tasks[task_name]['visual_prompt'] = visual_dict[task_name]

    return tasks


def add_audio_prompts(tasks, base_path, path_one, path_two):
    audio_base_dict = np.load(base_path)
    audio_one_dict = np.load(path_one)
    audio_two_dict = np.load(path_two)

    for task_name in tasks.keys():
        tasks[task_name]['audio_prompt_base'] = audio_base_dict[task_name]
        tasks[task_name]['audio_prompt_one'] = audio_one_dict[task_name]
        tasks[task_name]['audio_prompt_two'] = audio_two_dict[task_name]

    return tasks

# MineRL Environment ===================================================================================================


def load_agent_env(in_model, in_weights, task, seed, cond_scale):
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(task, seed)

    return agent, env


def make_env(task, seed=None):
    print('Loading MineRL...')
    env = EvaluationEnvironment(task, **ENV_KWARGS).make()
    print('Starting new env...')
    if seed is not None:
        print(f'Setting seed to {seed}...')
        env.seed(seed)
    env.reset()
    return env


class EvaluationEnvironment(HumanSurvival):
    def __init__(self, task, **kwargs):
        self.inventory = task.get('inventory', None)
        self.pref_biome = task.get('pref_biome', None)
        super().__init__(**kwargs)

    def create_agent_start(self) -> List[Handler]:
        retval = super().create_agent_start()
        if self.inventory is not None:
            retval.append(handlers.SimpleInventoryAgentStart(self.inventory))
        if self.pref_biome is not None:
            retval.append(handlers.PreferredSpawnBiome(self.pref_biome))
        return retval

# Calculate maximum cosine distance of episode to task =================================================================


def cosine_similarity(embedding1, embedding2):
    # normalizing
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)

    # cosine similarity
    similarity = (embedding1 @ embedding2.t()).cpu()

    return similarity


def eval_cosine_dist(task_embeding, video_path, metric_path, model):
    metrics = load_json(metric_path)
    sim = metrics.get('max_similarity', None)
    if sim is not None:
        print('Cosine similarity already computed - skipping')
        return
    video_dataset = VideoDataset(video_path, FPS, NUM_FRAMES)
    dataloader = DataLoader(video_dataset, batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=PREFETCH_FACTOR, num_workers=NUM_WORKERS)

    task_embedding = torch.tensor(task_embeding).to(torch.float32).to(DEVICE)
    result = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(DEVICE)
            embeddings = model.encode_video(batch)
            # if 'audio' in video_path:
            #    embeddings = model.project_video_embeddings(embeddings)
            result.append(embeddings)
    result = torch.cat(result)

    similarities = cosine_similarity(task_embedding, result)
    max_similarity = torch.max(similarities).item()
    min_similarity = torch.min(similarities).item()
    metrics['max_similarity'] = max_similarity
    save_json(metric_path, metrics)
    print(f'Maximal cosine distance: {max_similarity}, minimal cosine distance: {min_similarity}')

# Evaluator ============================================================================================================


class Evaluator:
    def __init__(self, initial_observation, task_name, task):
        self.init_obs = initial_observation
        self.metric = {'task_name': task_name, 'key': task['key'], 'count': 0}
        self.metric = compute_metrics(initial_observation, initial_observation, self.metric)

    def update(self, observation):
        self.metric = compute_metrics(self.init_obs, observation, self.metric)

    def evaluate(self):
        print('Evaluation of current episode:')
        count = self.metric['count']
        name = self.metric['task_name']
        print(f'{name}: {count}')

    def save(self, path):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        metric = self.metric
        metric['count'] = int(metric['count'])
        save_json(path, self.metric)


def compute_metrics(init_obs, curr_obs, metric):
    inventory = curr_obs['inventory']

    task_name = metric['task_name']
    if metric['key'] == 'dist':
        x_pos, y_pos = curr_obs['location_stats']['xpos'], curr_obs['location_stats']['zpos']
        start_x, start_y = init_obs['location_stats']['xpos'], init_obs['location_stats']['zpos']

        dist = np.sqrt((x_pos - start_x) ** 2 + (y_pos - start_y) ** 2)
        if dist > metric['count']:
            metric['count'] = dist
    else:
        block_name = metric['key']
        count = 0
        names = [block for block in inventory.keys() if block_name in block]
        for name in names:
            count += inventory.get(name, 0)
        if 'place' not in task_name and count > metric['count']:
            old_value = metric['count']
            print(f'Updating count for {block_name} from {old_value} to {count}')
            metric['count'] = count
        if 'place' in task_name and count < metric['count']:
            old_value = metric['count']
            print(f'Updating count for {block_name} from {old_value} to {count}')
            metric['count'] = count

    return metric
