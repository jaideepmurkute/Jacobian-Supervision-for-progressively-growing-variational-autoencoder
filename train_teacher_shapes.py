import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import argparse
from dataloader_shapes_new import Shapes_dataset
from new_arch_models import autoencoder
from utils import *
import copy
import seaborn as sns
from torchvision.utils import save_image
import argparse

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# ------------------------------------------ Set configuration -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('C', default=3, type=int, nargs='?', help='Number of channels in input image')
parser.add_argument('H', default=64, type=int, nargs='?', help='Height of the input image')
parser.add_argument('W', default=64, type=int, nargs='?', help='Width of the input image')
parser.add_argument('hidden_size', default=3, nargs='?', type=int, help='Size of the latent layer in VAE')
parser.add_argument('num_epochs', default=25, nargs='?', type=int, help='Number of epochs to train model')
parser.add_argument('batch_size', default=100, nargs='?', type=int, help='Batch size')
parser.add_argument('learning_rate', default=9, nargs='?', type=int, help='Learning rate')
parser.add_argument('l2_penalty', default=0.0, nargs='?', type=float, help='l2_penalty')
parser.add_argument('validate_during_training', nargs='?', default=False, type=bool, help='If True, Performance on validation set, '
                                                                               'will be computed, after each epoch')
parser.add_argument('load_from_checkpoint', default=False, nargs='?', type=bool, help='Load pre-trained model for further training')
args = parser.parse_args()

C = 3  # Number of input & output channels
H = 64  # Height of input
W = 64  # Width of input
input_size = H * W  # The image size = 64 x 64 = 4096
hidden_size = args.hidden_size  # The number of nodes at the hidden layer
num_epochs = 25  # The number of times entire dataset is trained
batch_size = 100  # changed from 128   # The size of input data took for one iteration
learning_rate = 5e-4  # The speed of convergence
l2_penalty = 0  # weight decay for optimizer
validate_during_training = False  # If True, prediction will be performed on the test set, after each epoch
load_from_checkpoint = False

BASE_SAVE_DIR = './shapes_new_arch_unsupervised'
model_name = 'unsupervised_teacher_x3_teach4'
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
        print("Starting Epoch: {}".format(epoch+1))
        epoch_train_loss = []
        epoch_recon_loss = []
        epoch_kld_loss = []
        count = 0
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            img, y = data
            img = img.view(img.size(0), -1)
            img = img.to(device)
            y = y.to(device)
            img = img.view(img.size(0), C, H, W)

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

            if batch_idx % 500 == 0:
                batch_log_string = "Batch: {} \t Train Loss: {:.8f} \t Recon. Loss: {:.8f} \t KLD Loss: {:.8f}" \
                    .format(batch_idx, loss.data.item(), reconstruction_loss.item(), kld_loss.item())
                print(batch_log_string)
                with open(training_log_file, 'a+') as fp:
                    fp.write(batch_log_string + '\n')
                plt.figure()
                npimg = img[0, :, :, :].cpu().detach().permute(1, 2, 0).numpy()
                plt.subplot(2, 1, 1)
                plt.imshow(npimg, interpolation='nearest', aspect='equal')
                plt.subplot(2, 1, 2)
                npimg_op = x_hat[0, :, :, :].cpu().detach().permute(1, 2, 0).numpy()
                plt.imshow(npimg_op, interpolation='nearest', aspect='equal')
                plt.savefig(SAMPLES_SAVE_DIR + '/epoch_' + str(epoch) + '_batch_' + str(batch_idx) + '_sample_0')
                plt.close()

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
                test_model(model, test_loader, hidden_size, 'whole', plot_latent_dist=False)
            print('Epoch [{}/{}], \t Total loss: {:.7f} \t Recon. Loss: {:.7f} \t KLD Loss: {:.7f}'
                  .format(epoch + 1, num_epochs, avg_total_loss, avg_recon_loss, avg_kld_loss))

        torch.save(model.state_dict(), MODEL_SAVE_DIR + '/' + model_name + '.pt')
        torch.save(optimizer.state_dict(), MODEL_SAVE_DIR + '/' + 'optimizer.pt')
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


def get_model(read_checkpoint=False, checkpoint_path=None, optimizer_checkpoint_path=None):
    model = autoencoder(hidden_size=hidden_size, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    if read_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Model loaded with weights at: ", checkpoint_path)
        if optimizer_checkpoint_path is not None:
            try:
                print("optimizer_checkpoint_path: ", optimizer_checkpoint_path)
                optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))
            except:
                print("Optimizer checkpoint not found ... Using new instace of optimizer ...")

    return model, optimizer


def get_data_loader(dataset_type, load_all_files=False):
    dataset_type = dataset_type.lower()
    is_test = (dataset_type == 'test')
    if dataset_type == 'whole_mig':
        load_all_files = True
    shuffle_flags = {'train': True, 'validation': False, 'test': False, 'whole_mig': False}
    print("shuffle_flags[dataset_type]: ", shuffle_flags[dataset_type])
    # dataset = Shapes_dataset(test=is_test, dir='data/', size=(H, W), all_files=load_all_files)
    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
    #                                           shuffle=shuffle_flags[dataset_type])
    # print("Number of batches per epoch: {}".format(len(data_loader)))
    dataset = Shapes_dataset(dir='./data', test=True, size=(H, W))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle_flags[dataset_type], num_workers=0,
                                              pin_memory=True)
    print("Number of batches per epoch: {}".format(len(data_loader)))
    return data_loader


def disentangle_check_image_row(model, data_loader, hidden_size):
  model.eval()
  imgs_save_dir = MODEL_SAVE_DIR+'/disentangle_img_row'
  gif_save_dir = MODEL_SAVE_DIR+'/disentangle_gif'
  if not os.path.exists(imgs_save_dir):
      os.mkdir(imgs_save_dir)
  if not os.path.exists(gif_save_dir):
      os.mkdir(gif_save_dir)

  image_shape = (64, 64, 3)
  with torch.no_grad():
    for data in data_loader:
      img, y = data
      img = img.to(device)
      x_hat_batch, z_sample_batch, mean_batch, sigma_batch = model(img)
      log_sigma_batch = torch.log(sigma_batch)
      break
  select_dim = []
  samples_allz = []
  z_mean = mean_batch[0].detach().cpu().numpy()
  z_sigma_sq = np.exp(log_sigma_batch[0].detach().cpu().numpy()) ** 2
  for ind in range(len(z_sigma_sq)):
    if z_sigma_sq[ind] < 0.2:
      select_dim.append(str(ind))

  plot_flag = True
  if plot_flag:
    n_z = z_mean.shape[0]
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
        rimg = reconstr_img[0, :, :, :].permute(1, 2, 0)
        samples.append(rimg)
      samples_allz.append(samples)
      imgs_comb = np.hstack((img.detach().cpu().numpy() for img in samples))
      image_path = imgs_save_dir+"/check_z{0}_{1}.png".format(target_z_index, 0)
      save_image(torch.from_numpy(imgs_comb).permute(2, 0, 1), image_path)
      samples = [samples[i].detach().cpu().numpy() for i in range(len(samples))]
      make_gif(samples, gif_save_dir+"/" + 'stu_latent' + "_z_%s.gif" % (target_z_index), duration = 2, true_image = False)
      print()
  final_gif = []
  for i in range(gif_nums + 1):
    gif_samples = []
    for j in range(hidden_size):
      gif_samples.append(samples_allz[j][i])
    imgs_comb = np.hstack((img.detach().cpu().numpy() for img in gif_samples))
    final_gif.append(imgs_comb)
  make_gif(final_gif, gif_save_dir+"/all_z_step{0}.gif".format(0), true_image=False)

  return select_dim


def test_model(model, data_loader, hidden_size, dataset_type, plot_latent_dist):
    model.eval()
    val_total_loss_log = []
    val_kld_loss_log = []
    val_recon_loss_log = []
    hidden_means_log = None
    hidden_sigma_log = None
    total_elbo_loss_value = 0
    total_kld_loss_value = 0
    total_recon_loss_value = 0
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
            total_elbo_loss_value += loss.item()
            total_kld_loss_value += kld_loss.item()
            total_recon_loss_value += reconstruction_loss.item()

            # if batch_idx == 3:
            #    break

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

    print("Avg. ELBO loss per sample: {:.5f}".format(total_elbo_loss_value))
    print("Avg. KLD loss per sample: {:.5f}".format(total_kld_loss_value))
    print("Avg. recon loss per sample: {:.5f}".format(total_recon_loss_value))

    return np.mean(val_total_loss_log), np.mean(val_recon_loss_log), np.mean(val_kld_loss_log), hidden_mean, \
           hidden_sigma


def traverse_latent_space(model, latent_size, n=5, gif_num=20, maxi=5, data_loader=None, dataset_type=None):
    # from unsupervised_disentangling_shapes.utils import imgs2gif

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
        train_loader = get_data_loader(dataset_type='train', load_all_files=True)
        z = mean.clone()
        z = z[:n, :]

        if not os.path.exists(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type)):
            os.mkdir(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type))

        for ri in range(gif_num + 1):
            value = -maxi + (2.0 * maxi / gif_num) * ri
            z[:, k] = value

            out = model.decode(z)
            singleImage = out.view(n, 3, 64, 64)

            singleImage = singleImage.cpu().detach()

            save_image(singleImage, PLOTS_SAVE_DIR + '/z_{}_{}/img_{}.png'.format(k, dataset_type, ri), nrow=n)
            # plt.figure()
            # plt.imshow(singleImage.numpy())
            # plt.savefig(PLOTS_SAVE_DIR+'/z_{}/img_{}.png'.format(k, ri))
            # plt.close()

            str_path = PLOTS_SAVE_DIR + '/z_{}_{}/'.format(k, dataset_type)

        imgs2gif(image_path=str_path, gif_path=PLOTS_SAVE_DIR, gif_name='gif_image_{}_{}'.format(k, dataset_type))


if __name__ == "__main__":
    choice = int(input("Enter Choice: 1] Train \t 2] Test"))
    optimizer = None
    if choice == 1:
        if load_from_checkpoint:
            model, optimizer = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR + '/' + model_name + '.pt',
                                         optimizer_checkpoint_path=MODEL_SAVE_DIR + '/optimizer.pt')
        else:
            model, optimizer = get_model(read_checkpoint=False)


        train_loader = get_data_loader(dataset_type='train', load_all_files=True)
        # from dsprites_prashnna import Dsprites_dataset
        # train_loader = Dsprites_dataset()
        # exit()

        test_loader = None
        if validate_during_training:
            test_loader = get_data_loader(dataset_type='test')
        model, losses, loss_names = train_model(model, train_loader, test_loader=test_loader,
                                                validate_during_training=validate_during_training)
        plot_training_history(losses, loss_names, PLOTS_SAVE_DIR, batch_size)
    elif choice == 2:
        print("passing: ", MODEL_SAVE_DIR + '/' + model_name)
        model, optimizer = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR + '/' + model_name + '.pt')

        data_loader = get_data_loader(dataset_type='whole_mig', load_all_files=True)

        avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
                  test_model(model, data_loader, hidden_size, 'whole_mig', plot_latent_dist=True)

        select_dim = disentangle_check_image_row(model, data_loader, hidden_size)
        print("select_dim: ", select_dim)
        print('-' * 50)
        
        print("Computing MIG metric ...")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(vae=model, shapes_dataset=None,
                                                                               dataset_loader=data_loader, nparams=2,
                                                                               K=hidden_size)
        print("MIG metric on {} dataset: {:.5f}".format('Whole', metric))
        
        del data_loader
        print("Computing Factor-VAE score ...")
        dataset = Shapes_dataset(dir='./data', test=True, size=(args.H, args.W))
        fac_metric = factor_metric_dsprite(dataset='3dshapes', dataset_reference=dataset)
        factor_vae_score = fac_metric.evaluate_mean_disentanglement(model)
        print("Mean Disentanglement Metric: " + str(factor_vae_score))

        disentangle_layer_sample(model, data_loader, args, PLOTS_SAVE_DIR, step=1)

    else:
        print("Entered choice: {}. Please enter valid option.".format(choice))