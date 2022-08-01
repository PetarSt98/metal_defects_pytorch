import config
from config import *
from model import Model
from dataLoader import CustomDataset

print('Path Direcorty: ', os.listdir(config.DATASET_DIR))
print("Train Direcorty: ", os.listdir(config.TRAIN_DIR))
print("Test Direcorty: ", os.listdir(config.TEST_DIR))
print("Validation Direcorty: ", os.listdir(config.VAL_DIR))

print('\n')
print("Training Inclusion data:", len(os.listdir(config.TRAIN_DIR + '/' + 'Inclusion')))
print("Testing Inclusion data:", len(os.listdir(config.TEST_DIR + '/' + 'Inclusion')))
print("Validation Inclusion data:", len(os.listdir(config.VAL_DIR + '/' + 'Inclusion')))

if __name__ == "__main__":
    train_dataset = CustomDataset(config.TRAIN_DIR, config.transformations)
    test_dataset = CustomDataset(config.TEST_DIR, config.transformations)
    val_dataset = CustomDataset(config.VAL_DIR, config.transformations)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Model().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WD)
    cumulative_loss = []
    cumulative_loss_val = []
    loss = 0
    loss_val = 0
    best_model = {'model': model, 'val_loss': np.inf}

    for epoch in range(config.EPOCHS):
        for images, labels in train_data_loader:
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for images, labels in val_data_loader:
            labels_pred = model(images)
            loss_val = criterion(labels_pred, labels)

        if loss_val < best_model['val_loss']:
            best_model = {'model': Model().to(config.DEVICE), 'val_loss': loss_val}
            best_model['model'].load_state_dict(model.state_dict())

        print(f'Epoch:{epoch + 1}, Loss_train:{loss.item():.4f}, , Loss_val:{loss_val.item():.4f}')

    print(f'Loading the best model with validation loss: {best_model["val_loss"]}')
    model = best_model['model']
    torch.save(model.state_dict(), config.SAVE_MODEL_PATH + 'best_model.pt')

    model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_data_loader:
            labels_pred = model(images)
            _, predicted = torch.max(labels_pred.data, 1)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on test dataset: {correct / test_dataset.__len__() * 100:.2f}%')
