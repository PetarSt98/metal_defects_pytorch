import config
from config import *
from model import Model
from dataLoader import CustomDataset


if __name__ == "__main__":
    test_dataset = CustomDataset(config.TEST_DIR, config.transformations)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Model().to(config.DEVICE)
    if len(config.LOAD_MODEL_PATH) > 0:
        model.load_state_dict(torch.load(config.LOAD_MODEL_PATH + 'best_model.pt'))

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        correct = 0
        for images, labels in test_data_loader:
            labels_pred = model(images)
            _, predicted = torch.max(labels_pred.data, 1)
            correct += (predicted == labels).sum().item()
            y_true.append(labels.to(torch.device('cpu')).numpy())
            y_pred.append(predicted.to(torch.device('cpu')).detach().numpy())

        print(f'Accuracy of the model on test dataset: {correct / test_dataset.__len__() * 100:.2f}%')

    y_true = np.concatenate(np.array(y_true))
    y_pred = np.concatenate(np.array(y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in config.LABELS],
                         columns=[i for i in config.LABELS])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    plt.show()