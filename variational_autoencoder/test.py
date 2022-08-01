import config
from config import *
from dataLoader import CustomDataset
from autoencoder import VAE
from dataLoader import CustomDataset


def final_loss(bce_loss, mu, log_var):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def vaildate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    recon_images = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total= int(len(dataloader)/config.BATCH_SIZE)):
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

    criterion = nn.BCELoss(reduction='sum')
    model = VAE(3, config.LATENT_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.LOAD_MODEL_PATH + 'best_model_vae.pt'))

    valid_epoch_loss, recon_images_batches = vaildate(model, train_data_loader, criterion)
    recon_images = recon_images_batches[0]
    imgs = recon_images.detach().cpu()

    for i in range(64):
        imgs[i] = CustomDataset.renormalize(imgs[i])
    imgs = imgs.numpy()
    cnt = 0
    fig, axs = plt.subplots(8, 8, figsize=(15, 15))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(np.reshape(imgs[cnt, :, ...], (64, 64, 3)))
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig('Augmented.png')
    fig.suptitle('AUGMENTED IMAGES')
    plt.show()

    imgs = next(iter(train_data_loader))[0].detach().cpu()
    for i in range(64):
        imgs[i] = CustomDataset.renormalize(imgs[i])
    imgs = imgs.numpy()
    cnt = 0
    fig, axs = plt.subplots(8, 8, figsize=(15, 15))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(np.reshape(imgs[cnt, :, ...], (64, 64, 3)))
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig('Original.png')
    fig.suptitle('ORIGINAL IMAGES')
    plt.show()