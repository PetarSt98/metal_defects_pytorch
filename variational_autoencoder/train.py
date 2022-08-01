import config
from config import *
from autoencoder import VAE
from dataLoader import CustomDataset


def final_loss(bce_loss, mu, log_var):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader) / config.BATCH_SIZE)):
        counter += 1
        img_batch = data[0]
        optimizer.zero_grad()
        reconstruction, mu, log_var = model(img_batch)
        bce_loss = criterion(reconstruction, img_batch)
        loss = final_loss(bce_loss, mu, log_var)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def vaildate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    recon_images = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader) / config.BATCH_SIZE)):
            counter += 1
            img_batch = data[0]
            reconstruction, mu, log_var = model(img_batch)
            bce_loss = criterion(reconstruction, img_batch)
            loss = final_loss(bce_loss, mu, log_var)
            running_loss += loss.item()
            recon_images.append(reconstruction)

    val_loss = running_loss / counter
    return val_loss, recon_images


if __name__ == '__main__':
    train_dataset = CustomDataset(config.TRAIN_DIR, config.transformations)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = VAE(3, config.LATENT_DIM).to(config.DEVICE)
    n_epochs = config.EPOCHS
    optimizer = optim.Adam(model.parameters(), lr=config.BATCH_SIZE)
    criterion = nn.BCELoss(reduction='sum')
    # criterion = nn.MSELoss()

    train_loss = []
    valid_loss = []
    for epoch in range(n_epochs):
        train_epoch_loss = train(model, train_data_loader, optimizer, criterion)
        valid_epoch_loss, _ = vaildate(model, train_data_loader, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        # clear_output(wait=True)
        print(f'Epoch:{epoch + 1}/{n_epochs} \t Train Loss: {train_epoch_loss:.4f} \t Val Loss: {valid_epoch_loss:.4f}')

    torch.save(model.state_dict(), config.SAVE_MODEL_PATH + 'best_model_vae.pt')
