# Imports
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from .load import load_model, save_model
from .preprocess import make_features


def train_mineclip_model(model,
                dataloader,
                savepath,
                cfg_path=None,
                param_path=None,
                lr=0.00025,
                test_size=1200,
                num_tests=10,
                start_epochs=0,     # Epochs without scheduler step
                num_epochs=100):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, str):
        chkpt = torch.load(model, map_location=device)
        model = load_model(chkpt, cfg_path, device)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.temporal_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(dataloading) // 2))
    # loss_fnc = nn.L1Loss()
    loss_fnc = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * 2)

    # Load parameters of previous training if training should be continued
    last_epoch = 0
    if param_path is not None:
        print("Continue training from checkpoint")
        dic = torch.load(param_path, map_location=device)
        last_epoch = dic['last_epoch']
        optimizer.load_state_dict(dic['optimizer'])
        train_losses = dic['train_losses']
        test_losses = dic['test_losses']
        train_accuracies = dic['train_accuracies']
        test_accuracies = dic['test_accuracies']
        if 'scheduler' in dic.keys():
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - last_epoch) * 2)
            scheduler.load_state_dict(dic['scheduler'])

    # training
    for epoch in range(last_epoch, num_epochs):
        model.train()
        dataloader.randomize_sample_order()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("Training...")
        for num_batch in tqdm(range(len(dataloader)), desc="Processing batch: "):
            batch_loss = train_batch(num_batch, model, dataloader, optimizer, loss_fnc, device)
            train_losses.append(batch_loss)

            # Testing
            optimizer.step()
        # Update scheduler once per epoch if startup epochs are done
        if epoch > start_epochs:
            scheduler.step()

        # Evaluate epoch performance
        print("Average training loss: {}".format(sum(train_losses[-len(dataloader):]) / len(dataloader)))

        loss, train_acc, test_acc = evaluate(model, dataloader, loss_fnc, test_size, num_tests, device)
        test_losses.append(loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Save progress every 100 epochs
        if (epoch + 1) % 100 == 0 and epoch != num_epochs - 1:
            save_model_params(model, optimizer,
                              train_losses, test_losses,
                              train_accuracies, test_accuracies,
                              last_epoch + epoch, savepath,
                              scheduler)

    print("Done training saving model")
    save_model_params(model, optimizer,
                      train_losses, test_losses,
                      train_accuracies, test_accuracies,
                      last_epoch + num_epochs, savepath,
                      scheduler)

    return model


def train_batch(batch_num, model, dataloader, optimizer, loss_fnc, device):
    optimizer.zero_grad()
    video_embeddings, audio_embeddings = dataloader.get_batch(batch_num)
    video_embeddings = torch.tensor(video_embeddings).to(device)
    audio_embeddings = torch.tensor(audio_embeddings).to(device)

    audio_embeds = model.project_audio_embeddings(audio_embeddings)

    loss = loss_fnc(audio_embeds, video_embeddings)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, dataloader, loss_fnc, test_size, num_tests, device):
    print("Validating...")
    model.eval()
    train_acc = 0
    loss = 0
    test_acc = 0

    for _ in range(num_tests):
        # Evaluate model performance in train set
        video_embeddings, audio_embeddings = dataloader.get_random_train_samples(test_size)
        video_embeddings = torch.tensor(video_embeddings).to(device)
        audio_embeddings = torch.tensor(audio_embeddings).to(device)
        acc = evaluate_model(model, video_embeddings, audio_embeddings)
        train_acc += acc

        # Evaluate model performance on evaluation set
        video_embeddings, audio_embeddings = dataloader.get_random_test_samples(test_size)
        video_embeddings = torch.tensor(video_embeddings).to(device)
        audio_embeddings = torch.tensor(audio_embeddings).to(device)

        # Loss of test set
        with torch.no_grad():
            audio_embeds = model.project_audio_embeddings(audio_embeddings)

        partial_loss = loss_fnc(audio_embeds, video_embeddings)
        loss += partial_loss.item()

        # Accuracy of test set
        acc = evaluate_model(model, video_embeddings, audio_embeddings)
        test_acc += acc

    # Average accuracies and test-loss
    train_acc /= num_tests
    loss /= num_tests
    test_acc /= num_tests
    print("  Average train accuracy:", train_acc)
    print("  Average test loss:", loss)
    print("  Average test accuracy:", test_acc)

    return loss, train_acc, test_acc


def evaluate_model(model, video_embeddings, audio_embeddings):
    with torch.no_grad():
        audio_encs = model.project_audio_embeddings(audio_embeddings)
        video_embeddings /= video_embeddings.norm(dim=-1, keepdim=True)
        audio_encs /= audio_encs.norm(dim=-1, keepdim=True)

    video_probs = (video_embeddings @ audio_encs.T).softmax(dim=-1).cpu()
    video_labels = torch.argmax(video_probs, dim=-1)
    labels = torch.arange(len(video_labels))
    correct_v = torch.sum(video_labels == labels).item()
    acc = correct_v / len(video_labels)

    return acc


def save_model_params(model,
                      optimizer,
                      train_losses,
                      test_losses,
                      train_accuracies,
                      test_accuracies,
                      epoch,
                      savepath,
                      scheduler=None):
    dir_path = os.path.dirname(savepath)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    name, extension = os.path.splitext(savepath)
    param_path = name + '_train_params' + extension
    if os.path.isfile(param_path):
        os.remove(param_path)
    if os.path.isfile(savepath):
        os.remove(savepath)
    dic = {
        'last_epoch': epoch,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'optimizer': optimizer.state_dict(),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }
    if scheduler is not None:
        dic['scheduler'] = scheduler.state_dict()
    # Save training parameters and model seperately
    torch.save(dic, param_path)
    save_model(model, savepath)
