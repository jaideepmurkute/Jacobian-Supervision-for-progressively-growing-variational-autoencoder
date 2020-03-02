import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import argparse
from unsupervised_disentangling_shapes.dataloader_shapes import Shapes_dataset
from unsupervised_disentangling_shapes.new_arch_models import mnist_autoencoder
from unsupervised_disentangling_shapes.utils import min_max_scaling, latent_loss, squared_error, compute_elbo_loss, make_gif
import copy
# from new_arch_models import autoencoder
# from utils import min_max_scaling, latent_loss, squared_error, compute_elbo_loss
import torchvision.datasets as dsets
import torchvision.transforms as transforms


import seaborn as sns
from torchvision.utils import save_image

sns.set(color_codes=True)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
params = parser.parse_args()

# ------------------------------------------ Set configuration -----------------------------------------------------
C = 1  # Number of input & output channels
H = 28  # Height of input
W = 28  # Width of input
input_size = H * W  # The image size = 64 x 64 = 4096
hidden_size = 3  # The number of nodes at the hidden layer
num_epochs = 20  # The number of times entire dataset is trained
batch_size = 512  # changed from 128   # The size of input data took for one iteration
learning_rate = 3e-4  # The speed of convergence
l2_penalty = 0  # weight decay for optimizer
validate_during_training = False  # If True, prediction will be performed on the test set, after each epoch
load_from_checkpoint = False

BASE_SAVE_DIR = './shapes_new_arch_unsupervised'
model_name = 'unsupervised_teacher_mnist1'
print("model_name: ", model_name)
MODEL_SAVE_DIR = BASE_SAVE_DIR + '/' + model_name
print("MODEL_SAVE_DIR: ", MODEL_SAVE_DIR)
# exit()
SAMPLES_SAVE_DIR = MODEL_SAVE_DIR + '/reconstructed_samples'
PLOTS_SAVE_DIR = MODEL_SAVE_DIR + '/plots'
training_log_file = MODEL_SAVE_DIR + '/training_log.txt'
try:
    os.remove(training_log_file)  # Delete the old log file, if exists
except OSError:
    pass

if not os.path.exists(BASE_SAVE_DIR):
    os.mkdir(BASE_SAVE_DIR)
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)
if not os.path.exists(SAMPLES_SAVE_DIR):
    os.mkdir(SAMPLES_SAVE_DIR)
if not os.path.exists(PLOTS_SAVE_DIR):
    os.mkdir(PLOTS_SAVE_DIR)


# ------------------------------------------------------------------------------------------------------------------


def train_model(model, train_loader, test_loader=None, validate_during_training=False):
    model.train()
    total_loss_log = []
    kld_loss_log = []
    recon_loss_log = []
    for epoch in range(num_epochs):
        print("Starting Epoch: {}".format(epoch))
        epoch_train_loss = []
        epoch_recon_loss = []
        epoch_kld_loss = []
        count = 0
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            img, y = data
            # img = img.view(img.size(0), -1)
            img = img.to(device)
            y = y.to(device)
            # img = img.view(img.size(0), C, H, W)

            x_hat, z_sample, z_mean, z_stddev = model(img)

            # loss = criterion(output, img)
            loss, reconstruction_loss, kld_loss = compute_elbo_loss(input=img, x_hat=x_hat, z_mean=z_mean,
                                                                    z_stddev=z_stddev)

            total_loss_log.append(loss.item())
            kld_loss_log.append(kld_loss.item())
            recon_loss_log.append(reconstruction_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.data.item())
            epoch_recon_loss.append(reconstruction_loss.item())
            epoch_kld_loss.append(kld_loss.item())

            if batch_idx % 2000 == 0:
                batch_log_string = "Batch: {} \t Train Loss: {:.8f} \t Recon. Loss: {:.8f} \t KLD Loss: {:.8f}" \
                    .format(batch_idx, loss.data.item(), reconstruction_loss.item(), kld_loss.item())
                print(batch_log_string)
                with open(training_log_file, 'a+') as fp:
                    fp.write(batch_log_string + '\n')
                # plt.figure()
                # npimg = img[0, :, :, :].cpu().detach().permute(1, 2, 0).numpy()
                # plt.subplot(2, 1, 1)
                # plt.imshow(npimg, interpolation='nearest', aspect='equal')
                # plt.subplot(2, 1, 2)
                # npimg_op = x_hat[0, :, :, :].cpu().detach().permute(1, 2, 0).numpy()
                # plt.imshow(npimg_op, interpolation='nearest', aspect='equal')
                # plt.savefig(SAMPLES_SAVE_DIR + '/epoch_' + str(epoch) + '_batch_' + str(batch_idx) + '_sample_0')
                # plt.close()

            # if batch_idx == 3:
            #     break

        # ===================log========================
        print("Epoch Average Summary on the training set: ")
        epoch_log_string = 'Epoch [{}/{}], \t Total loss: {:.7f} \t Recon. Loss: {:.7f} \t KLD Loss: {:.7f}'.format(
            epoch + 1,
            num_epochs, np.mean(epoch_train_loss), np.mean(epoch_recon_loss), np.mean(epoch_kld_loss))
        print(epoch_log_string)
        with open(training_log_file, 'a+') as fp:
            fp.write(epoch_log_string + '\n')
        if validate_during_training:
            print()
            print("Average performance on the test set: ")
            avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
                test_model(model, test_loader, hidden_size, dataset_type, plot_latent_dist=False)
            print('Epoch [{}/{}], \t Total loss: {:.7f} \t Recon. Loss: {:.7f} \t KLD Loss: {:.7f}'
                  .format(epoch + 1, num_epochs, avg_total_loss, avg_recon_loss, avg_kld_loss))

        torch.save(model.state_dict(), MODEL_SAVE_DIR + '/' + model_name + '.pt')
        print("-" * 100)

    total_loss_log = np.array(total_loss_log)
    recon_loss_log = np.array(recon_loss_log)
    kld_loss_log = np.array(kld_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_total_loss_log.npy', arr=total_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_recon_loss_log.npy', arr=recon_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_kld_loss_log.npy', arr=kld_loss_log)

    loss_names = ['ELBO Loss', 'Reconstruction Loss', 'KL-Divergence']
    losses = [total_loss_log, recon_loss_log, kld_loss_log]

    return model, losses, loss_names


def plot_training_history(losses, loss_names):
    num_losses = len(losses)
    # ------------- Unnormalized Loss Plots -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Unnormalized Loss Plots")
    for i in range(num_losses):
        plt.subplot(num_losses, 1, i + 1)
        plt.plot(losses[i])
        plt.ylabel(loss_names[i])
    plt.xlabel("Number of mini-batches with batch size: {}".format(batch_size))
    plt.savefig(PLOTS_SAVE_DIR + '/Unnormalized Loss Plots.png', dpi=500)
    plt.close()
    # ------------- Normalized Loss Plots -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Normalized Loss Plots")
    for i in range(num_losses):
        plt.subplot(num_losses, 1, i + 1)
        plt.plot(min_max_scaling(losses[i]))
        plt.ylabel(loss_names[i])
    plt.xlabel("Number of mini-batches with batch size: {}".format(batch_size))
    plt.savefig(PLOTS_SAVE_DIR + '/Normalized Loss Plots.png', dpi=500)
    plt.close()
    # ------------------------------------------------


def get_model(read_checkpoint=False, checkpoint_path=None):
    model = mnist_autoencoder(hidden_size=hidden_size, device=device).to(device)
    if read_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def get_data_loader(dataset_type):
    dataset_type = dataset_type.lower()
    is_test = (dataset_type == 'test')
    shuffle_flags = {'train': True, 'validation': False, 'test': False}
    is_train = not is_test
    dataset = dsets.MNIST(root='./data',
                                train=is_train,
                                transform=transforms.ToTensor(),
                                download=True)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle_flags[dataset_type])

    print("Number of batches per epoch: {}".format(len(data_loader)))

    return data_loader


def test_model(model, data_loader, hidden_size, dataset_type, plot_latent_dist):
    model.eval()
    val_total_loss_log = []
    val_kld_loss_log = []
    val_recon_loss_log = []
    hidden_means_log = None
    hidden_sigma_log = None
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            img, y = data
            img = img.view(img.size(0), -1)
            img = img.to(device)
            y = y.to(device)
            img = img.view(img.size(0), C, H, W)

            x_hat, z_sample, z_mean, z_stddev = model(img)
            if hidden_means_log is None:
                hidden_means_log = z_mean
                hidden_sigma_log = z_stddev
            else:
                hidden_means_log = torch.cat([hidden_means_log, z_mean], dim=0)
                hidden_sigma_log = torch.cat([hidden_sigma_log, z_stddev], dim=0)

            # loss = criterion(output, img)
            loss, reconstruction_loss, kld_loss = compute_elbo_loss(input=img, x_hat=x_hat, z_mean=z_mean,
                                                                    z_stddev=z_stddev)

            val_total_loss_log.append(loss.item())
            val_recon_loss_log.append(reconstruction_loss.item())
            val_kld_loss_log.append(kld_loss.item())

            # if batch_idx == 3:
            #     break

    np.save(file=MODEL_SAVE_DIR + '/test_total_loss_log.npy', arr=val_total_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/test_recon_loss_log.npy', arr=val_recon_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/test_kld_loss_log.npy', arr=val_kld_loss_log)

    hidden_means_log = hidden_means_log.cpu().numpy()
    hidden_sigma_log = hidden_sigma_log.cpu().numpy()

    if plot_latent_dist:
        plt.figure(figsize=(15, 12))
        for i in range(hidden_size):
            plt.subplot(hidden_size, 1, i + 1)
            sns.kdeplot(hidden_means_log[:, i], shade=True, color="b")
            plt.ylabel("Latent Factor: " + str(i + 1))
        # plt.tight_layout()
        plt.title("KDE plot for the latent distributions - " + dataset_type)
        plt.savefig(MODEL_SAVE_DIR + '/latent_distribution_' + str(dataset_type), dpi=500)
        plt.close()

    # average mean and sigma for each latent factor
    hidden_mean = np.mean(hidden_means_log, axis=0)
    hidden_sigma = np.mean(hidden_sigma_log, axis=0)


    return np.mean(val_total_loss_log), np.mean(val_recon_loss_log), np.mean(val_kld_loss_log), hidden_mean, \
           hidden_sigma


def traverse_latent_space(model, latent_size, n=5, gif_num=20, maxi=5, data_loader=None, dataset_type=None):
    from unsupervised_disentangling_shapes.utils import imgs2gif
    # from utils import imgs2gif
    gif_num = 20  # how many pics in the gif
    maxi = 5.0

    for iter, batch in enumerate(data_loader):
        # print(batch)
        batch, y = batch
        batch = batch.to(device)  # only fine resolution image
        # x_hat, z_sample, mean, self.sigma
        # mu, logvar, x_recon = model(batch)
        x_hat, z_sample, mean, sigma = model(batch)
        break

    for k in range(latent_size):
        z = mean.clone()
        z = z[:n, :]

        if not os.path.exists(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type)):
            os.mkdir(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type))

        for ri in range(gif_num + 1):
            value = -maxi + (2.0 * maxi / gif_num) * ri
            z[:, k] = value

            out = model.decode(z)
            singleImage = out.view(n, C, H, W)

            singleImage = singleImage.cpu().detach()

            save_image(singleImage, PLOTS_SAVE_DIR + '/z_{}_{}/img_{}.png'.format(k, dataset_type, ri), nrow=n)
            # plt.figure()
            # plt.imshow(singleImage.numpy())
            # plt.savefig(PLOTS_SAVE_DIR+'/z_{}/img_{}.png'.format(k, ri))
            # plt.close()

            str_path = PLOTS_SAVE_DIR + '/z_{}_{}/'.format(k, dataset_type)

        imgs2gif(image_path=str_path, gif_path=PLOTS_SAVE_DIR, gif_name='gif_image_{}_{}'.format(k, dataset_type))


def disentangle_check_image_row(model, data_loader, hidden_size):
  model.eval()
  imgs_save_dir = MODEL_SAVE_DIR+'/disentangle_img_row'
  gif_save_dir = MODEL_SAVE_DIR+'/disentangle_gif'
  if not os.path.exists(imgs_save_dir):
      os.mkdir(imgs_save_dir)
  if not os.path.exists(gif_save_dir):
      os.mkdir(gif_save_dir)

  image_shape = (28, 28, 3)
  # get random mini-batch
  with torch.no_grad():
    for data in data_loader:
      img, y = data
      img = img.to(device)
      x_hat_batch, z_sample_batch, mean_batch, log_sigma_batch = model(img)
      break
  # recon = x_hat_batch[0].detach().cpu().permute(1, 2, 0).numpy()
  # plt.figure()
  # plt.imshow(recon)
  # plt.show()
  # print("z_sample_batch.size(): ", z_sample_batch.size())
  select_dim = []
  samples_allz = []
  # print("mean_batch.size(): ", mean_batch.size())
  z_mean = mean_batch[0].detach().cpu().numpy()
  # print("z_mean.shape: ", z_mean.shape)
  z_sigma_sq = np.exp(log_sigma_batch[0].detach().cpu().numpy())
  # print("z_sigma_sq.shape: ", z_sigma_sq.shape)
  for ind in range(len(z_sigma_sq)):
    if z_sigma_sq[ind] < 0.2:
      select_dim.append(str(ind))

  plot_flag = True
  if plot_flag:
    n_z = z_mean.shape[0]
    # print("z_mean.shape: ", z_mean.shape)
    # print("n_z: ", n_z)
    for target_z_index in range(n_z):
      print("Traversing latent unit : ", target_z_index)
      samples = []
      gif_nums = 20
      for ri in range(gif_nums + 1):
        # value = -3.0 + (6.0 / 9.0) * ri
        maxi = 3
        value = -maxi + 2 * maxi / gif_nums * ri
        code2 = copy.deepcopy(z_sample_batch.detach().cpu().numpy())
        for i in range(n_z):
          if i == target_z_index:
            code2[0][i] = value
          else:
            code2[0][i] = code2[0][i]
        reconstr_img = model.decode(torch.from_numpy(code2).cuda())
        # reconstr_img = model.decode(z_sample_batch)
        # print("reconstr_img.size(): ", reconstr_img.size())
        rimg = reconstr_img[0, :, :, :].permute(1, 2, 0)
        # print("rimg.size(): ", rimg.size())
        # print("rimg: ", rimg)
        # print("rimg * 255: ", rimg * 255)
        samples.append(rimg)
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(samples[-1].detach().cpu().numpy(), interpolation='nearest', aspect='equal')
        # plt.subplot(2, 1, 2)
        # plt.imshow(samples[-1].detach().cpu().numpy()*255, interpolation='nearest', aspect='equal')
        # plt.show()
        # plt.close()
      # print("len(samples) i.e. # samples for current latent unit: ", len(samples))
      samples_allz.append(samples)
      imgs_comb = np.hstack((img.detach().cpu().numpy() for img in samples))
      print("imgs_comb.shape: ", imgs_comb.shape)
      # print("imgs_comb.shape: ", imgs_comb.shape)
      # plt.imshow(imgs_comb)
      # plt.show()
      # plt.close()
      image_path = imgs_save_dir+"/check_z{0}_{1}.png".format(target_z_index, 0)
      save_image(torch.from_numpy(imgs_comb).permute(2, 0, 1), image_path)
      # image_path, gif_path='', gif_name='gif_image', gif_duration=0.035
      # imgs2gif(image_path="./disentangle_img_row/", gif_path="./disentangle_img_row_gif/", gif_name="gif_".format(target_z_index))
      samples = [samples[i].detach().cpu().numpy() for i in range(len(samples))]
      make_gif(samples, gif_save_dir+"/" + 'stu_latent' + "_z_%s.gif" % (target_z_index), duration = 2, true_image = False)
      print()
  print("samples_allz.shape: ", np.array(samples_allz).shape)
  final_gif = []
  for i in range(gif_nums + 1):
    gif_samples = []
    for j in range(hidden_size):
      gif_samples.append(samples_allz[j][i])
    imgs_comb = np.hstack((img.detach().cpu().numpy() for img in gif_samples))
    final_gif.append(imgs_comb)
  # image_path, gif_path='', gif_name='gif_image', gif_duration=0.035
  # imgs2gif(image_path="./disentangle_img_row/", gif_path="./disentangle_img_row_gif/", gif_name="second_gif_".format(target_z_index))
  make_gif(final_gif, gif_save_dir+"/all_z_step{0}.gif".format(0), true_image=False)

  return select_dim

if __name__ == "__main__":
    choice = int(input("Enter Choice: 1] Train \t 2] Test"))

    if choice == 1:
        if load_from_checkpoint:
            model = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR + '/' + model_name + '.pt')
        else:
            model = get_model(read_checkpoint=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

        train_loader = get_data_loader(dataset_type='train')
        test_loader = None
        if validate_during_training:
            test_loader = get_data_loader(dataset_type='test')
        model, losses, loss_names = train_model(model, train_loader, test_loader=test_loader,
                                                validate_during_training=validate_during_training)
        plot_training_history(losses, loss_names)
    elif choice == 2:
        print("passing: ", MODEL_SAVE_DIR + '/' + model_name)
        model = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR + '/' + model_name + '.pt')

        # dataset_type = 'train'
        # data_loader = get_data_loader(dataset_type=dataset_type)
        # avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
        #     test_model(model, data_loader, hidden_size, dataset_type, plot_latent_dist=True)
        # print("Loss values for the train dataset: ")
        # print(
        #     "Total Loss: {:.5f} \t Recon. Loss: {:.5f} \t KL Divergence: {:.5f}".format(avg_total_loss, avg_recon_loss,
        #                                                                                 avg_kld_loss))
        # print("Train set - Latent Mean: ", hidden_mean)
        # print("Train set - Latent Sigma: ", hidden_sigma)
        # traverse_latent_space(model, n=5, gif_num=20, maxi=5, latent_size=hidden_size, data_loader=data_loader,
        #                       dataset_type='train')
        # del data_loader
        # print()
        # dataset_type = 'test'
        # data_loader = get_data_loader(dataset_type=dataset_type)
        # avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
        #     test_model(model, data_loader, hidden_size, dataset_type, plot_latent_dist=True)
        # print("Loss values for the test dataset: ")
        # print(
        #     "Total Loss: {:.5f} \t Recon. Loss: {:.5f} \t KL Divergence: {:.5f}".format(avg_total_loss, avg_recon_loss,
        #                                                                                 avg_kld_loss))
        # print("Test set - Latent Mean: ", hidden_mean)
        # print("Test set - Latent Sigma: ", hidden_sigma)
        # traverse_latent_space(model, n=5, gif_num=20, maxi=5, latent_size=hidden_size, data_loader=data_loader,
        #                       dataset_type='test')

        data_loader = get_data_loader(dataset_type='train')
        select_dim = disentangle_check_image_row(model, data_loader, hidden_size)
        print("select_dim: ", select_dim)
        print('-' * 50)



    else:
        print("Entered choice: {}. Please enter valid option.".format(choice))