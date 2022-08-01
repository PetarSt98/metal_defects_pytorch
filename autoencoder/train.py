import config
from config import *
from dataLoader import CustomDataset
from autoencoder import Autoencoder, weights_init


def train(model_ae, optimizer_ae, criterion_ae, scheduler_ae):
    print("Starting Training Loop...")
    for epoch in range(config.EPOCHS):
        for data, labels in train_data_loader:
            optimizer_ae.zero_grad()
            recon = model_ae(data)
            loss_ae = criterion_ae(recon, data)
            loss_ae.backward()
            optimizer_ae.step()
        print(f'Epochs:{epoch} \t Loss: {loss_ae.item()}')
        scheduler_ae.step(loss_ae)

    torch.save(model_ae.state_dict(), config.SAVE_MODEL_PATH + 'best_model_ae.pt')


if __name__ == '__main__':
    train_dataset = CustomDataset(config.TRAIN_DIR, config.transformations)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model_ae = Autoencoder().to(config.DEVICE)
    criterion_ae = nn.MSELoss().to(config.DEVICE)
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=config.LR, weight_decay=config.WD)
    scheduler_ae = ReduceLROnPlateau(optimizer_ae, 'min', factor=0.5, patience=5, verbose=True)

    model_ae.apply(weights_init)

    train(model_ae, optimizer_ae, criterion_ae, scheduler_ae)
