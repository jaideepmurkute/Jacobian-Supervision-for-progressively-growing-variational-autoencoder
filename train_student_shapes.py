import matplotlib.pyplot as plt
from dataloader_shapes_new import Shapes_dataset
from new_arch_models import autoencoder
from utils import *
from elbo_decomposition import *
import seaborn as sns
import copy
import argparse
from torchvision.utils import save_image
import sys
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# ------------------------------------------ Set configuration -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('C', default=3, nargs='?', type=int, help='Number of channels in input image')
parser.add_argument('H', default=64, nargs='?', type=int, help='Height of the input image')
parser.add_argument('W', default=64, nargs='?', type=int, help='Width of the input image')
parser.add_argument('hidden_size', default=9, nargs='?', type=int, help='Size of the latent layer in VAE')
parser.add_argument('teacher_hidden_size', default=6, nargs='?', type=int, help='Size of the latent '
                                                                                'layer in teacher VAE')
parser.add_argument('num_epochs', default=23, nargs='?', type=int, help='Number of epochs to train model')
parser.add_argument('batch_size', default=100, nargs='?', type=int, help='Batch size')
parser.add_argument('learning_rate', default=5e-4, nargs='?', type=int, help='Learning rate')
parser.add_argument('l2_penalty', default=0.0, nargs='?', type=float, help='l2_penalty')
parser.add_argument('validate_during_training', default=False, nargs='?', type=bool,
                    help='If True, Performance on validation set, will be computed, after each epoch')
parser.add_argument('load_from_checkpoint', default=True, nargs='?', type=bool,
                    help='Load pretrained model for further training')
parser.add_argument('use_jacobian_supervision', default=True, nargs='?', type=bool,
                    help='To enable the Jacobian supervision')
parser.add_argument('lambda_z', default=1.0, nargs='?', type=float,
                    help='Scalar for the factor-retention loss in the Jacobian supervision')
parser.add_argument('lambda_jacobian', default=1.0, nargs='?', type=float,
                    help='Scalar for the Jacobian loss in the Jacobian supervision')
parser.add_argument('lambda_xcov', default=0.0001, nargs='?', type=float,
                    help='Scalar for the cross-covariance loss in the Jacobian supervision')
parser.add_argument('model_name', type=str, nargs='?', help='Name of the student model')
parser.add_argument('teacher_model_name', type=str, nargs='?', help='Name of the teacher model')
args = parser.parse_args()
print("model_names from args: ", args.model_name)
print("teacher_model_name from args: ", args.teacher_model_name)

C = 3  # Number of input & output channels
H = 64  # Height of input
W = 64  # Width of input
input_size = H * W  # The image size = 64 x 64 = 4096
hidden_size = args.hidden_size  # The number of nodes at the hidden layer
teacher_hidden_size = args.teacher_hidden_size
num_epochs = 25  # The number of times entire dataset is trained
batch_size = 100  # changed from 128   # The size of input data took for one iteration
learning_rate = 5e-4  # The speed of convergence
l2_penalty = 0.  # weight decay for optimizer
validate_during_training = False  # If True, prediction will be performed on the test set, after each epoch
load_from_checkpoint = False
use_jacobian_supervision = True
lambda_z = 1.0
lambda_jacobian = 1.0
lambda_xcov = 0.0001

BASE_SAVE_DIR = './shapes_new_arch_unsupervised'
model_name = 'unsupervised_student_x3_stud8_jac'  # Student model name
teacher_model_name = 'unsupervised_student_x3_stud7_jac'
print("Student model name: ", model_name)
print("Teacher model name: ", teacher_model_name)
MODEL_SAVE_DIR = BASE_SAVE_DIR + '/' + model_name
SAMPLES_SAVE_DIR = MODEL_SAVE_DIR+'/reconstructed_samples'
PLOTS_SAVE_DIR = MODEL_SAVE_DIR + '/plots'
training_log_file = MODEL_SAVE_DIR + '/training_log.txt'

if not os.path.exists(BASE_SAVE_DIR):
    os.mkdir(BASE_SAVE_DIR)
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)
if not os.path.exists(SAMPLES_SAVE_DIR):
    os.mkdir(SAMPLES_SAVE_DIR)
if not os.path.exists(PLOTS_SAVE_DIR):
    os.mkdir(PLOTS_SAVE_DIR)
# ------------------------------------------------------------------------------------------------------------------


#  Define Cross-covariance loss """
def compute_xcov(z, y, bs):
    """
    computes cross-covariance loss between latent code and attributes
    prediction, so that latent code does note encode attributes, compute
    mean first.
    """
    # z: latent code
    # y: predicted labels
    # bs: batch size
    z = z.contiguous().view(bs, -1)
    y = y.contiguous().view(bs, -1)

    # center matrices
    z = z - torch.mean(z, dim=0)
    y = y - torch.mean(y, dim=0)

    cov_matrix = torch.matmul(torch.t(z), y)

    cov_loss = torch.norm(cov_matrix.view(1, -1)) / bs

    return cov_loss



def train_model(model, train_loader, test_loader=None, validate_during_training=False, teacher_model=None,
                dataset_type=None):
    model.train()
    # model.double()
    total_loss_log = []
    kld_loss_log = []
    recon_loss_log = []
    factor_retention_loss_log = []
    xcov_loss_log = []
    jacobian_loss_log = []
    min_loss = sys.maxsize
    for epoch in range(args.num_epochs):
        print("Starting Epoch: {}".format(epoch+1))
        epoch_train_loss = []
        epoch_recon_loss = []
        epoch_kld_loss = []
        epoch_factor_retention_loss = []
        epoch_xcov_loss = []
        epoch_jacobian_loss = []
        count = 0
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            img, y = data
            # img = img.double()
            img = img.view(img.size(0), -1)
            img = img.to(device)
            img = img.view(img.size(0), args.C, args.H, args.W)

            x_hat, z_sample, z_mean, z_stddev = model(img)

            loss, reconstruction_loss, kld_loss = compute_elbo_loss(input=img, x_hat=x_hat, z_mean=z_mean,
                                                                    z_stddev=z_stddev)
            # print("after batch {} : Recon Loss isnan: {} KLD loss isnan: {}".format(batch_idx,
            #                                                                         torch.isnan(reconstruction_loss).any().item(),
            #                                                                         torch.isnan(kld_loss).any().item()))
            if args.use_jacobian_supervision:
                factor_retention_loss, xcov_loss, jacobian_loss = get_other_losses(model, teacher_model, img,
                                                               x_hat, z_sample, args.lambda_z, args.lambda_xcov,
                                                                                   args.lambda_jacobian)

                loss += factor_retention_loss
                loss += xcov_loss
                loss += jacobian_loss

                factor_retention_loss_log.append(factor_retention_loss.item())
                xcov_loss_log.append(xcov_loss.item())
                jacobian_loss_log.append(jacobian_loss.item())
                epoch_factor_retention_loss.append(factor_retention_loss.item())
                epoch_xcov_loss.append(xcov_loss.item())
                epoch_jacobian_loss.append(jacobian_loss.item())

            total_loss_log.append(loss.item())
            kld_loss_log.append(kld_loss.item())
            recon_loss_log.append(reconstruction_loss.item())
            epoch_train_loss.append(loss.data.item())
            epoch_recon_loss.append(reconstruction_loss.item())
            epoch_kld_loss.append(kld_loss.item())

            # if torch.isnan(loss).any().item() > 0:
            #     print("before batch {} loss has NaN value ...".format(batch_idx))
            #     exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if torch.isnan(loss).any().item() > 0:
            #     print("after batch {} loss has NaN value ...".format(batch_idx))
            #     exit()

            if batch_idx % 500 == 0:
                batch_log_string = "Batch: {} \t Train Loss: {:.8f} \t Recon. Loss: {:.8f} \t KLD Loss: {:.8f}"\
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
                plt.savefig(SAMPLES_SAVE_DIR+'/epoch_' + str(epoch)+'_batch_'+str(batch_idx)+'_sample_0')
                plt.close()

            # if batch_idx == 3:
            #     break

        # ===================log========================
        print("Epoch Average Summary on the training set: ")
        epoch_log_string = 'Epoch [{}/{}], \t Total loss: {:.7f} \t Recon. Loss: {:.7f} \t KLD Loss: {:.7f}'.format(epoch + 1,
                            args.num_epochs, np.mean(epoch_train_loss), np.mean(epoch_recon_loss), np.mean(epoch_kld_loss))
        print(epoch_log_string)
        with open(training_log_file, 'a+') as fp:
            fp.write(epoch_log_string+'\n')
        if args.validate_during_training:
            print()
            print("Average performance on the test set: ")
            avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
                test_model(model, test_loader, args.hidden_size, dataset_type, plot_latent_dist=False)
            print('Epoch [{}/{}], \t Total loss: {:.7f} \t Recon. Loss: {:.7f} \t KLD Loss: {:.7f}'
                  .format(epoch + 1, args.num_epochs, avg_total_loss, avg_recon_loss, avg_kld_loss))

        torch.save(model.state_dict(), MODEL_SAVE_DIR+'/' + model_name + '.pt')
        if epoch == 15:
            torch.save(model.state_dict(), MODEL_SAVE_DIR + '/' + model_name + '_epoch15.pt')
        if loss.item() < min_loss:
            print("New best epoch found ... Saving model ...")
            torch.save(model.state_dict(), MODEL_SAVE_DIR + '/' + model_name + '_epoch_best.pt')
            min_loss = loss.item()
        print("-"*100)

    total_loss_log = np.array(total_loss_log)
    recon_loss_log = np.array(recon_loss_log)
    kld_loss_log = np.array(kld_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_total_loss_log.npy', arr=total_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_recon_loss_log.npy', arr=recon_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/train_kld_loss_log.npy', arr=kld_loss_log)

    loss_names = ['ELBO Loss', 'Reconstruction Loss', 'KL-Divergence']
    losses = [total_loss_log, recon_loss_log, kld_loss_log]

    return model, losses, loss_names


def get_model(read_checkpoint=False, checkpoint_path=None, teacher_path=None):
    student_model = autoencoder(hidden_size=args.hidden_size, device=device).to(device)
    teacher_model = autoencoder(hidden_size=args.teacher_hidden_size, device=device).to(device)
    teacher_model.load_state_dict(torch.load(teacher_path))
    print("teacher model weights loaded from: ", teacher_path)
    if read_checkpoint:
        print("student checkpoint path: ", checkpoint_path)
        print("teacher checkpoint path: ", teacher_path)
        student_model.load_state_dict(torch.load(checkpoint_path))
    else:
        """# Copy parameters from teacher to student"""
        for k in student_model.state_dict().keys():
            if student_model.state_dict()[k].size() != teacher_model.state_dict()[k].size():
                # copy manually
                their_model = teacher_model.state_dict()[k]
                my_model = student_model.state_dict()[k]
                sz = their_model.size()
                if k.startswith('compute'):
                    print("1..Copied the unequal sized layer weights, at layer: {}".format(k))
                    print("my_model.size(): ", my_model.size())
                    print("their_model.size(): ", their_model.size())

                    if torch.isnan(their_model[:]).any().item() > 0:
                        print("teacher model has NaN weight value ...")
                        exit()
                    my_model[-sz[0]:] = their_model[:]
                else:
                    print("2..Copied the unequal sized layer weights, at layer: {}".format(k))
                    print("my_model.size(): ", my_model.size())
                    print("their_model.size(): ", their_model.size())
                    if torch.isnan(their_model[:]).any().item() > 0:
                        print("teacher model has NaN weight value ...")
                        exit()
                    my_model[:, -sz[1]:] = their_model[:]

            else:
                # copy straight
                print("Copied all the weights from {} layer of teacher to student".format(k))
                student_model.state_dict()[k].copy_(teacher_model.state_dict()[k])

        print("Student Loaded with teacher weights from: ", teacher_path)

    return student_model, teacher_model


def get_data_loader(dataset_type, load_all_files=False):
    dataset_type = dataset_type.lower()
    shuffle_flags = {'train': True, 'validation': False, 'test': False, 'whole_mig': False}
    print("shuffle_flags[dataset_type] is: ", shuffle_flags[dataset_type])

    dataset = Shapes_dataset(dir='./data', test=True, size=(args.H, args.W))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=shuffle_flags[dataset_type], num_workers=0,
                                              pin_memory=True)
    print("Number of batches per epoch: {}".format(len(data_loader)))

    return data_loader


def test_model(model, data_loader, hidden_size, dataset_type, plot_latent_dist, teacher_model):
    model.eval()
    val_total_loss_log = []
    val_kld_loss_log = []
    val_recon_loss_log = []
    hidden_means_log = None
    hidden_sigma_log = None
    print("Testing model ...")
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            img, y = data
            img = img.view(img.size(0), -1)
            img = img.to(device)
            img = img.view(img.size(0), args.C, args.H, args.W)

            x_hat, z_sample, z_mean, z_stddev = model(img)
            if hidden_means_log is None:
                hidden_means_log = z_mean
                hidden_sigma_log = z_stddev
            else:
                hidden_means_log = torch.cat([hidden_means_log, z_mean], dim=0)
                hidden_sigma_log = torch.cat([hidden_sigma_log, z_stddev], dim=0)

            loss, reconstruction_loss, kld_loss = compute_elbo_loss(input=img, x_hat=x_hat, z_mean=z_mean,
                                                                    z_stddev=z_stddev)

            val_total_loss_log.append(loss.item())
            val_recon_loss_log.append(reconstruction_loss.item())
            val_kld_loss_log.append(kld_loss.item())

    np.save(file=MODEL_SAVE_DIR + '/test_total_loss_log.npy', arr=val_total_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/test_recon_loss_log.npy', arr=val_recon_loss_log)
    np.save(file=MODEL_SAVE_DIR + '/test_kld_loss_log.npy', arr=val_kld_loss_log)

    hidden_means_log = hidden_means_log.cpu().numpy()
    hidden_sigma_log = hidden_sigma_log.cpu().numpy()

    if plot_latent_dist:
        plt.figure(figsize=(15, 12))
        for i in range(args.hidden_size):
            plt.subplot(args.hidden_size, 1, i+1)
            sns.kdeplot(hidden_means_log[:, i], shade=True, color="b")
            plt.ylabel("Latent Factor: "+str(i+1))
        # plt.tight_layout()
        plt.title("KDE plot for the latent distributions - " + dataset_type)
        plt.savefig(MODEL_SAVE_DIR + '/latent_distribution_'+str(dataset_type), dpi=500)
        plt.close()

    # average mean and sigma for each latent factor
    hidden_mean = np.mean(hidden_means_log, axis=0)
    hidden_sigma = np.mean(hidden_sigma_log, axis=0)

    return np.mean(val_total_loss_log), np.mean(val_recon_loss_log), np.mean(val_kld_loss_log), hidden_mean, \
           hidden_sigma


def disentangle_check_image_row(model, data_loader, hidden_size):
    model.eval()
    imgs_save_dir = MODEL_SAVE_DIR + '/disentangle_img_row'
    gif_save_dir = MODEL_SAVE_DIR + '/disentangle_gif'
    if not os.path.exists(imgs_save_dir):
        os.mkdir(imgs_save_dir)
    if not os.path.exists(gif_save_dir):
        os.mkdir(gif_save_dir)

    image_shape = (args.H, args.W, args.C)
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
                rimg = reconstr_img[0, :, :, :].permute(1, 2, 0)  # change this line for dsprites, to have (64, 64) output - without channels
                samples.append(rimg*255)
            samples_allz.append(samples)
            imgs_comb = np.hstack((img.detach().cpu().numpy() for img in samples))
            image_path = imgs_save_dir + "/check_z{0}_{1}.png".format(target_z_index, 0)
            save_image(torch.from_numpy(imgs_comb).permute(2, 0, 1), image_path)
            samples = [samples[i].detach().cpu().numpy() for i in range(len(samples))]
            make_gif(samples, gif_save_dir + "/" + 'stu_latent' + "_z_%s.gif" % (target_z_index), duration=2,
                     true_image=False)
            print()
    final_gif = []
    for i in range(gif_nums + 1):
        gif_samples = []
        for j in range(args.hidden_size):
            gif_samples.append(samples_allz[j][i])
        imgs_comb = np.hstack((img.detach().cpu().numpy() for img in gif_samples))
        final_gif.append(imgs_comb)
    make_gif(final_gif, gif_save_dir + "/all_z_step{0}.gif".format(0), true_image=False)

    return select_dim


def evaluate_model(model, data_loader):
    avg_total_loss, avg_recon_loss, avg_kld_loss, hidden_mean, hidden_sigma = \
        test_model(model, data_loader, args.hidden_size, '', plot_latent_dist=True, teacher_model=teacher_model)
    print("Loss values for the {} dataset: ")
    print("Total Loss: {:.5f} \t Recon. Loss: {:.5f} \t KL Divergence: {:.5f}".format(avg_total_loss, avg_recon_loss,
                                                                                      avg_kld_loss))
    print('-' * 50)

    print("Test set - Latent Mean: ", hidden_mean)
    print("Test set - Latent Sigma: ", hidden_sigma)


if __name__ == "__main__":
    choice = int(input("Enter Choice: 1] Train \t 2] Test"))
    teacher_path = BASE_SAVE_DIR + '/' + teacher_model_name + '/' + teacher_model_name+'.pt'
    if choice == 1:
        try:
            os.remove(training_log_file)  # Delete the old log file, if exists
        except OSError:
            pass

        if args.load_from_checkpoint:
            model, teacher_model = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR+'/' + model_name + '.pt',
                              teacher_path=teacher_path)
        else:
            model, teacher_model = get_model(read_checkpoint=False, teacher_path=teacher_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty)
        train_loader = get_data_loader(dataset_type='train')
        test_loader = None
        if args.validate_during_training:
            test_loader = get_data_loader(dataset_type='test')
        model, losses, loss_names = train_model(model, train_loader, test_loader=test_loader,
                                                validate_during_training=args.validate_during_training,
                                                teacher_model=teacher_model, dataset_type='train')
        plot_training_history(losses, loss_names, PLOTS_SAVE_DIR, args.batch_size)
        
    elif choice == 2:
        print("passing: ", MODEL_SAVE_DIR+'/' + model_name)
        print("student checkpoint path: ", MODEL_SAVE_DIR+'/' + model_name + '.pt')
        print("teacher checkpoint path: ", teacher_path)

        model, teacher_model = get_model(read_checkpoint=True, checkpoint_path=MODEL_SAVE_DIR+'/' + model_name + '.pt',
                                         teacher_path=teacher_path)

        print("Evaluating model performance ,,,")
        data_loader = get_data_loader(dataset_type='whole_mig', load_all_files=True)
        evaluate_model(model=model, data_loader=data_loader)
        print('-' * 50)


        print("Computing MIG metric ...")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(vae=model, shapes_dataset=None,
                                                                                  dataset_loader=data_loader, nparams=2,
                                                                                  K=args.hidden_size)
        print("MIG metric on {} dataset: {:.5f}".format('Whole', metric))
        
        print("Traversing latent space for visualization ...")
        # for i in range(100):
        select_dim = disentangle_check_image_row_dsprite_z(model, data_loader, args, PLOTS_SAVE_DIR,
                                                           dataset_type='3dshapes', run=24553)
        print("select_dim: ", select_dim)
        print('-'*50)

        # del data_loader
        #print("Computing Factor-VAE score ...")
        #dataset = Shapes_dataset(dir='./data', test=True, size=(args.H, args.W))
        #fac_metric = factor_metric_dsprite(dataset='3dshapes', dataset_reference=dataset)
        #factor_vae_score = fac_metric.evaluate_mean_disentanglement(model)
        #print("Mean Disentanglement Metric: " + str(factor_vae_score))

        # print("Computing MI change between teacher and student ...")
        # logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        #     elbo_decomposition(model, teacher_model, data_loader, args)

        # for i in range(100):
        # disentangle_layer_sample(model, data_loader, args, PLOTS_SAVE_DIR, step=1, run=24553)


    else:
        print("Entered choice: {}. Please enter valid option.".format(choice))