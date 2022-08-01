import config
from config import *
from GAN_models import Discriminator, Generator
from utils import initialize_weights, gradient_penalty


def train(gen, disc, opt_gen, opt_disc, writer_real, writer_fake):
    step = 0
    gen.train()
    disc.train()

    for epoch in range(config.EPOCHS):
        for batch_idx, (real, labels) in enumerate(train_data_loader):
            real = real.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            current_batch_size = real.shape[0]

            for _ in range(config.CRITIC_ITERATIONS):
                noise = torch.randn(current_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
                fake = gen(noise, labels)
                critic_real = disc(real, labels).reshape(-1)
                critic_fake = disc(fake, labels).reshape(-1)
                gp = gradient_penalty(disc, labels, real, fake)
                loss_disc = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)
                disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                opt_disc.step()

            gen_fake = disc(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 5 == 0 and batch_idx > 0:
                print(f'Epoch [{epoch}/{config.EPOCHS}] Batch {batch_idx}/{len(train_data_loader)} \
                Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}')

            with torch.no_grad():
                fake = gen(noise, labels)

                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image('Real', img_grid_real, global_step=step)
                writer_fake.add_image('Fake', img_grid_fake, global_step=step)

            step += 1


if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(config.TRAIN_DIR, transform=config.transformations)
    train_data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    gen = Generator(
        config.Z_DIM,
        config.CHANNELS_NUM,
        config.FEATURES_GEN,
        len(config.LABELS),
        config.IMAGE_SIZE,
        config.GEN_EMBEDDING
    ).to(config.DEVICE)

    disc = Discriminator(
        config.CHANNELS_NUM,
        config.FEATURES_DISC,
        len(config.LABELS),
        config.IMAGE_SIZE
    ).to(config.DEVICE)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LR, betas=(config.BETA1, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LR, betas=(config.BETA1, 0.99))

    writer_real = SummaryWriter(f'logs/GAN/real')
    writer_fake = SummaryWriter(f'logs/GAN/fake')

    train(gen, disc, opt_gen, opt_disc, writer_real, writer_fake)
