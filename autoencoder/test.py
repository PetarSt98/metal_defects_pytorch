import config
from config import *
from dataLoader import CustomDataset
from autoencoder import Autoencoder, add_noise


def test(train_data_loader, model_ae):
    for data, labels in train_data_loader:
        encoded, ind1, ind2 = model_ae.encoder(data)
        encoded_noise = add_noise(encoded)
        new_data = model_ae.decoder(encoded_noise, ind2, ind1)

        fig, axs = plt.subplots(8, 8, figsize=(30, 30))
        cnt = 0
        for i in range(8):
            for j in range(8):
                axs[i, j].imshow(np.reshape(data.cpu()[cnt, :, ...].numpy(), (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
                axs[i, j].axis('off')
                cnt += 1
        plt.savefig('Original.png')
        plt.show()

        fig, axs = plt.subplots(8, 8, figsize=(30, 30))
        cnt = 0
        for i in range(8):
            for j in range(8):
                axs[i, j].imshow(np.reshape(new_data.detach().cpu()[cnt, :, ...].numpy(), (config.IMAGE_SIZE,
                                                                                           config.IMAGE_SIZE, 3)))
                axs[i, j].axis('off')
                cnt += 1
        plt.savefig('Augmented.png')
        plt.show()
        break


if __name__ == '__main__':
    train_dataset = CustomDataset(config.TRAIN_DIR, config.transformations)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model_ae = Autoencoder().to(config.DEVICE)
    model_ae.load_state_dict(torch.load(config.LOAD_MODEL_PATH + 'best_model_ae.pt'))

    test(train_data_loader, model_ae)
