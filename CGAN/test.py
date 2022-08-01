import config
from config import *
from dataLoader import CustomDataset
from GAN_models import Discriminator, Generator


def validate(gen, labels):
    labels = torch.tensor(labels).to(config.DEVICE)
    noise = torch.randn(len(labels), config.Z_DIM, 1, 1).to(config.DEVICE)
    fake = gen(noise, labels)
    fake = CustomDataset.renormalize(fake)

    return fake


if __name__ == '__main__':
    gen_test = Generator(config.Z_DIM, config.CHANNELS_NUM, config.FEATURES_GEN, len(config.LABELS), config.IMAGE_SIZE,
                         config.GEN_EMBEDDING).to(config.DEVICE)
    gen_test.load_state_dict(torch.load(config.LOAD_MODEL_PATH + 'best_model_gen.pt'))

    labels = [0, 1, 2, 3, 4, 5]
    fake = validate(gen_test, labels)

    fig, axs = plt.subplots(2, 3, figsize=(6, 6))
    cnt = 0
    for i in range(2):
        for j in range(3):
            axs[i, j].imshow(np.transpose(fake.detach().cpu()[cnt, :, ...].numpy()))
            axs[i, j].axis('off')
            axs[i, j].set_title(config.LABELS[labels[cnt]])
            cnt += 1
    plt.tight_layout()
    plt.savefig('Fake_images1.png')
    plt.show()