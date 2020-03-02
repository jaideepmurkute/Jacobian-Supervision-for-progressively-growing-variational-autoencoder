import torch
import torch.nn as nn
import numpy as np
import os
import re
import imageio
from torch.autograd import Variable
import math
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from numbers import Number
import dist as dist
import moviepy.editor as mpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import h5py
from scipy.misc import imsave
import copy
import matplotlib.pyplot as plt


def disentangle_layer_sample(model, data_loader, args, PLOTS_SAVE_DIR, step=1, run=0):
    # test_img_ind = 312015
    test_img_ind = 24553
    with torch.no_grad():
        inputs, labels = data_loader.dataset[test_img_ind]

    # qz = model.inference(img)
    # mean, log_sigma = model.encode_for_MIG_new(inputs)
    inputs = inputs.unsqueeze(0).cuda()
    x_hat, z_sample, mean, sigma = model(inputs)

    code = torch.Tensor(1, args.hidden_size)
    code[0, :] = z_sample
    # code[:, 1] = log_sigma

    num_rows = int(args.hidden_size / 3)
    print("num_rows: ", num_rows)
    num_cols = 6  # Number of samples
    randomize_slice = 3
    # if args.hidden_size == 3 or args.hidden_size == 6:
    #     randomize_upto_index = 3
    # if args.hidden_size == 9:
    #     randomize_upto_index = 6
    # print("Randomizing {}/{} units.".format(randomize_upto_index, args.hidden_size))

    image_comb_row = []
    randomize_start_index = 0
    for img_i in range(num_rows):
        samples = []
        randomize_end_index = randomize_start_index + randomize_slice

        for img_j in range(num_cols):
            code = code.detach()
            code2 = copy.deepcopy(code)

            print("randomize_start_index: ", randomize_start_index)
            print("randomize_end_index: ", randomize_end_index)

            code2[0, randomize_start_index:randomize_end_index] = \
                torch.from_numpy(np.random.normal(0, 1.5, randomize_slice).reshape(1, randomize_slice))
            reconstr_img = model.decode(code2.cuda()).detach().cpu()
            reconstr_img = reconstr_img.squeeze(0)

            reconstr_img = reconstr_img.permute(1, 2, 0).numpy()
            # rimg = reconstr_img[0].reshape((args.H, args.W, args.C)).astype(np.uint8)
            rimg = reconstr_img

            # plt.imshow(rimg*255)
            # plt.show()
            # plt.close()
            samples.append(rimg * 255)

        randomize_start_index = randomize_end_index

        imgs_comb = np.hstack((img for img in samples))
        image_comb_row.append(imgs_comb)
    final_comb = np.vstack((img for img in reversed(image_comb_row)))
    # imsave("disentangle_img_row/step_"+str(step)+"_disentangle_" + key + "_seed"+str(test_img_ind)+".png", final_comb)
    print("final_comb.shape: ", final_comb.shape)
    imsave(PLOTS_SAVE_DIR + "/step_" + str(step) + "_disentangle_seed" + str(run) + ".png",
           final_comb)
    # imageio.imwrite(PLOTS_SAVE_DIR + "/step_" + str(step) + "_disentangle_seed" + str(0) + ".png", final_comb)


def disentangle_check_image_row_dsprite_z(model, data_loader, args, PLOTS_SAVE_DIR, dataset_type='3dshapes', run=0):
    dataset_type = dataset_type.lower()

    if not os.path.exists(PLOTS_SAVE_DIR+"/disentangle_img"):
        os.mkdir(PLOTS_SAVE_DIR+"/disentangle_img")
    if not os.path.exists(PLOTS_SAVE_DIR+"/disentangle_img_row"):
        os.mkdir(PLOTS_SAVE_DIR+"/disentangle_img_row")

    model.eval()

    # with torch.no_grad():
    #     for data in data_loader:
    #         if dataset_type == 'dsprites':
    #             img = data
    #             img = img.unsqueeze(1)
    #         elif dataset_type == '3dshapes':
    #             img, y = data
    #
    #         img = img.cuda()
    #         x_hat_batch, z_sample_batch, mean_batch, sigma_batch = model(img)
    #         log_sigma_batch = torch.log(sigma_batch)
    #         break

    if dataset_type == '3dshapes':
        # test_img_ind = np.random.randint(480000 - 1)
        # test_img_ind = 312014
        test_img_ind = 24553
    elif dataset_type == 'dsprites':
        test_img_ind = 0
    with torch.no_grad():
        data = data_loader.dataset[test_img_ind]
        if dataset_type == 'dsprites':
            inputs = data
            inputs = inputs.unsqueeze(0)
        else:
            inputs, labels = data
        inputs_np = inputs
        inputs_np = inputs_np.permute(1, 2, 0)
        # print("inputs_np.shape: ", inputs_np.shape)
        inputs_np = inputs_np.numpy()
        plt.imshow(inputs_np)
        plt.savefig(PLOTS_SAVE_DIR+'/disentangle_img_row/image_'+str(test_img_ind))
        plt.close()
        # exit()
        inputs = inputs.unsqueeze(0)
        x_hat_batch, z_sample_batch, mean_batch, sigma_batch = model(inputs.cuda())
        log_sigma_batch = torch.log(sigma_batch)

    image_shape = (args.H, args.W, args.C)
    # qz = model.inference(sess, img)
    # code = model.inference_z(sess, img)
    code = z_sample_batch
    select_dim = []
    samples_allz = []
    z_mean = mean_batch
    # z_sigma_sq = np.pow(qz[key][1][0])
    # print("sigma_batch[1][0]: ", sigma_batch[1][0])
    # print("code.size(): ", code.size())
    # print("sigma_batch.size(): ", sigma_batch.size())
    z_sigma_sq = (sigma_batch[0] ** 2).cpu().numpy()

    for ind in range(len(z_sigma_sq)):
        if z_sigma_sq[ind] < 0.2:
            select_dim.append(str(ind))
    # print("z_mean.shape[0]: ", z_mean.shape[0])
    gif_nums = 9
    n_z = z_mean.shape[1]
    collage = None

    if dataset_type == '3dshapes':
        # collage = np.zeros((n_z*args.H, args.W*gif_nums, args.C))
        collage = np.zeros((3 * args.H, args.W * gif_nums, args.C))
    else:
        collage = np.zeros((n_z * args.H, args.W * gif_nums))

    # maxi = 0.8
    part_index = 0
    mult_index = 0
    for target_z_index in range(n_z):
        print("target_z_index: ", target_z_index)
        samples = []
        mean = 0
        # random_vals = np.random.normal(loc=mean, scale=maxi, size=gif_nums)
        # random_vals.sort()
        for ri in range(gif_nums):
            # value = -3.0 + (6.0 / 9.0) * ri

            # value = random_vals[ri]
            maxi = 2.5
            value = -maxi + 2 * maxi / gif_nums * ri
            # print("ri: {} value: {:.5f}".format(ri, value))
            code2 = copy.deepcopy(code)
            for i in range(n_z):
                if i == target_z_index:
                    code2[0][i] = value
                else:
                    code2[0][i] = code[0][i]
            # reconstr_img = model.generate(sess, code2)
            reconstr_img = model.decode(code2)
            # rimg = reconstr_img[0].reshape(image_shape).detach().cpu().numpy()
            #print("before squeezing reconstr_img[0].size(): ", reconstr_img[0].size())
            rimg = reconstr_img[0].squeeze(0)
            if dataset_type == '3dshapes':
                rimg = rimg.permute(1, 2, 0).detach().cpu().numpy()
            else:
                rimg = rimg.detach().cpu().numpy()
            if dataset_type == 'dsprites':
                rimg = np.reshape(rimg, newshape=(args.H, args.W)) # , args.H*(gif_nums + 1)
            samples.append(rimg * 255)
        samples_allz.append(samples)
        imgs_comb = np.hstack((img for img in reversed(samples)))
        # samples.reverse()  # in-place reversal


        # imgs_comb = np.reshape(imgs_comb, newshape=(args.H, args.H*(gif_nums + 1)))

        # if dataset_type == 'dsprites':
        #     imsave(PLOTS_SAVE_DIR+"/disentangle_img_row/check_" + str(run) + "_z{0}.png".format(target_z_index),
        #            imgs_comb.astype(np.uint8))
        # elif dataset_type == '3dshapes':
        #     imageio.imwrite(PLOTS_SAVE_DIR+"/disentangle_img_row/check_" + str(run) + "_z{0}.png".format(target_z_index),
        #                     imgs_comb.astype(np.uint8))
        # plt.imshow(imgs_comb.astype(np.uint8))
        # plt.show()
        #make_gif(samples, PLOTS_SAVE_DIR+"/disentangle_img/" + "_z_%s.gif" % (target_z_index), true_image=True)

        if target_z_index in [3, 6] and dataset_type == '3dshapes':
            mult_index = 0

        start_height = collage.shape[0] - args.H * (mult_index + 1)
        mult_index += 1
        if target_z_index in [3, 6] and dataset_type == '3dshapes':
            part_index += 1
            start_height = collage.shape[0] - args.H

        end_height = start_height + args.H

        if dataset_type == '3dshapes':
            print("start_height: ", start_height)
            print("end_height: ", end_height)
            collage[start_height:end_height, :, :] = imgs_comb
        else:
            collage[start_height:end_height, :] = imgs_comb
        if target_z_index in [2, 5, 8] and dataset_type == '3dshapes':
            imageio.imwrite(PLOTS_SAVE_DIR+"/disentangle_img_row/collage_"+str(test_img_ind)+"_"+str(target_z_index)+".png", collage.astype(np.uint8))
            collage = np.zeros((3 * args.H, args.W * gif_nums, args.C))
            print("New collage created ...")
        elif dataset_type == 'dsprites':
            imageio.imwrite(PLOTS_SAVE_DIR + "/disentangle_img_row/collage_" + str(test_img_ind) + "_" + str(
                target_z_index) + ".png", collage.astype(np.uint8))

    # final_gif = []
    # for i in range(gif_nums):
    #     gif_samples = []
    #     for j in range(args.hidden_size):
    #         gif_samples.append(samples_allz[j][i])
    #     gif_samples.reverse()  # now it is ladder high to lader low
    #     imgs_comb = np.hstack((img for img in gif_samples))
    #     final_gif.append(imgs_comb)
    # make_gif(final_gif, PLOTS_SAVE_DIR+"/disentangle_img/all_z.gif", true_image=True)
    # make_gif(final_gif, PLOTS_SAVE_DIR+"/all_z.gif", true_image=True)

    return select_dim


def disentangle_check_image_row_shapes(model, data_loader, plot_flag=True, step=0):
    C = 3
    H = 64
    W = 64
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            img, y = data
            img = img.view(img.size(0), -1)
            img = img.cuda()
            img = img.view(img.size(0), C, H, W)
            x_hat, z_sample, z_mean, z_stddev = model(img)

    select_dim = []
    samples_allz = []
    z_mean = qz[key][0][0]
    z_sigma_sq = np.exp(qz[key][1][0])

    for ind in range(len(z_sigma_sq)):
        if z_sigma_sq[ind] < 0.2:
            select_dim.append(key + "_" + str(ind))

    gif_nums = 10
    if plot_flag:
        n_z = z_mean.shape[0]
        for target_z_index in range(n_z):
            samples = []
            for ri in range(gif_nums + 1):
                # value = -3.0 + (6.0 / 9.0) * ri
                maxi = 2.5
                value = -maxi + 2 * maxi / gif_nums * ri
                code2 = copy.deepcopy(code)
                for i in range(n_z):
                    if i == target_z_index:
                        code2[key][0][i] = value
                    else:
                        code2[key][0][i] = code[key][0][i]
                reconstr_img = model.generate(sess, code2)
                rimg = reconstr_img[0].reshape(image_shape)
                samples.append(rimg * 255)
            samples_allz.append(samples)
            imgs_comb = np.hstack((img for img in samples))
            imsave("disentangle_img_row/check_" + key + "_z{0}_{1}.png".format(target_z_index, test_img_ind),
                   imgs_comb)
            make_gif(samples, "disentangle_img/" + key + "_z_%s.gif" % (target_z_index), true_image=True)
    if plot_flag:
        final_gif = []
        for i in range(gif_nums):
            gif_samples = []
            for j in range(z_dim * len(qz.keys())):
                gif_samples.append(samples_allz[j][i])
            gif_samples.reverse()  # now it is ladder high to lader low
            imgs_comb = np.hstack((img for img in gif_samples))
            final_gif.append(imgs_comb)
        make_gif(final_gif, "disentangle_img/all_z_step{0}.gif".format(step), true_image=True)
        make_gif(final_gif, flags.checkpoint_dir + "/all_z_step{0}.gif".format(step), true_image=True)

    return select_dim

# def make_gif(images, fname, duration=2, true_image=False):
#   def make_frame(t):
#     try:
#       x = images[int(len(images)/duration*t)]
#     except:
#       x = images[-1]
#     print("x.shape: ", x.shape)
#     if true_image:
#       return x.astype(np.uint8)
#     else:
#       return ((x+1)/2*255).astype(np.uint8)
#
#   clip = mpy.VideoClip(make_frame, duration=duration)
#   clip.write_gif(fname, fps = len(images) / duration)


def min_max_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-16)


def get_kl_divergence(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev

    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def get_reconstruction_error(output, input):
    # print("Check nan values in output: ", torch.isnan(output).any())
    # print("Check nan values in input: ", torch.isnan(input).any())
    # if torch.isnan(output).any().item() != 0:
    #     # print("NaN values found in the output ... Replacing by 255")
    #     # print("output before: ", output)
    #     old_output = output
    #     output[output != output] = 1
    #     # print("output after: ", output)

    return F.binary_cross_entropy(output, input, reduction='sum')


def compute_elbo_loss(input, x_hat, z_mean, z_stddev):
    x_hat = torch.clamp(x_hat, min=0, max=1)
    input = torch.clamp(input, min=0, max=1)
    reconstruction_loss = get_reconstruction_error(output=x_hat, input=input)
    kld_loss = get_kl_divergence(z_mean=z_mean, z_stddev=z_stddev)
    loss = reconstruction_loss + kld_loss

    return loss, reconstruction_loss, kld_loss


""" #  Define Cross-covariance loss """
def compute_xcov(z,y,bs):
    """computes cross-covariance loss between latent code and attributes
    prediction, so that latent code does note encode attributes, compute
    mean first."""
    # z: latent code
    # y: predicted labels
    # bs: batch size
    z = z.contiguous().view(bs,-1)
    y = y.contiguous().view(bs,-1)

    # center matrices
    z = z - torch.mean(z, dim=0)
    y = y - torch.mean(y, dim=0)

    cov_matrix = torch.matmul(torch.t(z),y)
    cov_loss = torch.norm(cov_matrix.view(1,-1))/bs

    return cov_loss


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        # print("x.shape: ", x.shape)
        # plt.imshow(x)
        # plt.show()
        # print("true_image: ", true_image)
        # true_image = True
        # print("x: ", x)
        # print("x.astype(np.uint8): ", x.astype(np.uint8))
        if true_image:
            # plt.imshow(x.astype(np.uint8))
            # plt.show()
            return x.astype(np.uint8)
        else:
            plt.imshow(((x + 1) / 2  * 255).astype(np.uint8))
            plt.show()
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def imgs2gif(image_path, gif_path='', gif_name='gif_image', gif_duration=0.035):
    ''' Creats .gif images from bunch of images.

    Args:
        image_path (str): Directory that contains the images.
        gif_path (str): Directory that the output .gif will be saved.
        gif_name (str): Name of the output .gif.
        gif_duration (float): The duration (in sec) between images.

    Returns:
        An image type '.gif' of inputed images.
    '''
    # Get the image folder path
    # image_folder = os.fsencode(image_path)
    print("image_path: ", image_path)
    # print("image_folder: ", image_folder)
    image_folder = image_path
    # Read files
    filenames = []
    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.jpeg', '.png', '.gif')):
            filenames.append(image_path+filename)

    # Sort the files
    # -- Because the names are str with char and numbers,
    #    we need to construct the sort function.
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]
    filenames.sort(key=natural_keys)

    # Get the images
    images = list(map(lambda filename: imageio.imread(filename), filenames))

    # Build the .gif image and save it
    print("saving gif at: ", os.path.join(gif_path+gif_name+'.gif'))
    imageio.mimsave(os.path.join(gif_path+'/'+gif_name+'.gif'), images, duration=gif_duration)


def plot_training_history(losses, loss_names, PLOTS_SAVE_DIR, batch_size):
    num_losses = len(losses)
    # ------------- Unnormalized Loss Plots -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Unnormalized Loss Plots")
    for i in range(num_losses):
        plt.subplot(num_losses, 1, i+1)
        plt.plot(losses[i])
        plt.ylabel(loss_names[i])
    plt.xlabel("Number of mini-batches with batch size: {}".format(batch_size))
    plt.savefig(PLOTS_SAVE_DIR+'/Unnormalized Loss Plots.png', dpi=500)
    plt.close()
    # ------------- Normalized Loss Plots -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Normalized Loss Plots")
    for i in range(num_losses):
        plt.subplot(num_losses, 1, i+1)
        plt.plot(min_max_scaling(losses[i]))
        plt.ylabel(loss_names[i])
    plt.xlabel("Number of mini-batches with batch size: {}".format(batch_size, batch_size))
    plt.savefig(PLOTS_SAVE_DIR+'/Normalized Loss Plots.png', dpi=500)
    plt.close()
    # ------------------------------------------------



# def traverse_latent_space(model, latent_size, n=5, gif_num=20, maxi=5, data_loader=None, dataset_type=None, PLOTS_SAVE_DIR=None):
#     gif_num = 20  # how many pics in the gif
#     maxi = 5.0
#
#     for iter, batch in enumerate(data_loader):
#         batch, y = batch
#         batch = batch.to(device)  # only fine resolution image
#         x_hat, z_sample, mean, sigma = model(batch)
#         break
#
#     for k in range(latent_size):
#         z = mean.clone()
#         z = z[:n, :]
#
#         if not os.path.exists(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type)):
#             os.mkdir(PLOTS_SAVE_DIR + '/z_{}_{}'.format(k, dataset_type))
#
#         for ri in range(gif_num + 1):
#             value = -maxi + (2.0 * maxi / gif_num) * ri
#             z[:, k] = value
#
#             out = model.decode(z)
#             singleImage = out.view(n, 3, 64, 64)
#
#             singleImage = singleImage.cpu().detach()
#
#             save_image(singleImage, PLOTS_SAVE_DIR + '/z_{}_{}/img_{}.png'.format(k, dataset_type, ri), nrow=n)
#             str_path = PLOTS_SAVE_DIR + '/z_{}_{}/'.format(k, dataset_type)
#
#         imgs2gif(image_path=str_path, gif_path=PLOTS_SAVE_DIR, gif_name='gif_image_{}_{}'.format(k, dataset_type))
#


def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
    metric_name = 'MIG'
    # factor_entropies = [6, 40, 32, 32]
    factor_entropies = [10, 10, 10, 8, 4, 15]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric

def compute_metric_with_axis_shapes(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [10, 10, 10, 8, 4, 15]
    mutual_infos = marginal_entropies[None] - cond_entropies  # [4,6]
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_axis = eval('MIG')(mutual_infos_s2[active_units,:])
    return metric, metric_axis

def compute_metric_dsprite(marginal_entropies, cond_entropies):
    metric_name = 'MIG'
    factor_entropies = [6, 40, 32, 32]
    # factor_entropies = [10, 10, 10, 8, 4, 15]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric

# change this function a lot, can directly copy
def compute_metric_with_axis_dsprite(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies  # [4,6]
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_axis = eval('MIG')(mutual_infos_s2[active_units, :])
    return metric, metric_axis  # first one is MIG, second one is MIG_axis


def logsumexp(value, dim=None, keepdim=False):
    """
    Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None, nparams_model=2):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).
    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)
    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """
    # Only take a sample subset of the samples
    if weights is None:
        # input, dim, index, out=None
        qz_samples = qz_samples.index_select(dim=1,
                                             index=Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()

    N, _, nparams = qz_params.size()
    assert(nparams == nparams_model)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(20, S - k)  # changed from 10 to 20
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    entropies /= S

    return entropies


def mutual_info_metric_shapes(vae=None, shapes_dataset=None, dataset_loader=None, nparams=2, K=None):
    # dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)
    import copy
    N = len(dataset_loader.dataset)  # number of data samples
    vae.eval()
    q_dist = dist.Normal()
    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)
    qz_samples = torch.Tensor(N, K)
    n = 0
    with torch.no_grad():
        for data in tqdm(dataset_loader):
            xs, y = data
            batch_size = xs.size(0)
            xs = xs.view(batch_size, 3, 64, 64).cuda()
            mean, log_sigma = vae.encode_for_MIG_new(xs)  # (batch_size, z_dim, 2)
            qz_params[n:n + batch_size, :, 0] = mean
            qz_params[n:n + batch_size, :, 1] = log_sigma

            n += batch_size

    qz_params = Variable(qz_params.view(10, 10, 10, 8, 4, 15, K, nparams).cuda())  # 3d shapes
    qz_samples = dist.Normal().sample(params=qz_params)  # Same as re-parameterization, but explicit sampling -
                                                         # since we don't need to backprop
    qz_samples = qz_samples.cuda()  # (10, 10, 10, 8, 4, 15, K)
    n_samples = 10000

    # --------------------attention: need this block to compute active units and pass it to compute MIG later
    qz_means = qz_params[:, :, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > 1e-2].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, K))
    # --------------------------------------------------------------------------------------


    print('Estimating marginal entropies.')
    # marginal entropies
    # qz_samples, qz_params, q_dist, n_samples=10000, weights=None, nparams_model=2
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        q_dist=dist.Normal(), n_samples=n_samples)

    marginal_entropies = marginal_entropies.cpu()

    num_factors_of_variation = 6
    floor_hue_configs = 10
    wall_hue_configs = 10
    object_hue_configs = 10
    scale_configs = 8
    shape_configs = 4
    orientation_configs = 15
    cond_entropies = torch.zeros(num_factors_of_variation, K)

    qz_samples = qz_samples.view(10, 10, 10, 8, 4, 15, K)

    print('Estimating conditional entropies for floor-hue.')
    print("qz_samples.size(): ", qz_samples.size())
    for i in range(floor_hue_configs):
        qz_samples_scale = qz_samples[i, :, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // floor_hue_configs, K).transpose(0, 1),
            qz_params_scale.view(N // floor_hue_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[0] += cond_entropies_i.cpu() / floor_hue_configs

    print('Estimating conditional entropies for wall_hue_configs.')  # ------------
    for i in range(wall_hue_configs):
        qz_samples_scale = qz_samples[:, i, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // wall_hue_configs, K).transpose(0, 1),
            qz_params_scale.view(N // wall_hue_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[1] += cond_entropies_i.cpu() / wall_hue_configs

    print('Estimating conditional entropies for object_hue_configs.')
    for i in range(object_hue_configs):
        qz_samples_scale = qz_samples[:, :, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // object_hue_configs, K).transpose(0, 1),
            qz_params_scale.view(N // object_hue_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[2] += cond_entropies_i.cpu() / object_hue_configs

    print('Estimating conditional entropies for scale_configs.')
    for i in range(scale_configs):
        qz_samples_scale = qz_samples[:, :, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // scale_configs, K).transpose(0, 1),
            qz_params_scale.view(N // scale_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[3] += cond_entropies_i.cpu() / scale_configs

    print('Estimating conditional entropies for shape_configs.')
    for i in range(shape_configs):
        qz_samples_scale = qz_samples[:, :, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // shape_configs, K).transpose(0, 1),
            qz_params_scale.view(N // shape_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[4] += cond_entropies_i.cpu() / shape_configs

    print('Estimating conditional entropies for orientation_configs.')
    for i in range(orientation_configs):
        qz_samples_scale = qz_samples[:, :, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // orientation_configs, K).transpose(0, 1),
            qz_params_scale.view(N // orientation_configs, K, nparams),
            q_dist=dist.Normal(), n_samples=n_samples)

        cond_entropies[5] += cond_entropies_i.cpu() / orientation_configs

    # metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    metric, metric_axis = compute_metric_with_axis_shapes(marginal_entropies, cond_entropies, active_units)

    print("MIG Metric: ", metric)
    print("MIG_axis Metric: ", metric_axis)
    print("Marginal_entropies: ", marginal_entropies)
    print("Cond_entropies: ", cond_entropies)

    return metric, marginal_entropies, cond_entropies


class factor_metric_dsprite(object):
    def __init__(self, dataset='dsprite', dataset_reference=None):
        # Data
        dataset = dataset.lower()
        self.dataset_reference = dataset_reference
        self.data_file = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.data_train, self.data_test, self.all_imgs, self.all_factors, self.n_classes = self._data_init(dataset=dataset)

    def _data_init(self, dataset):
        # Find dataset here: https://github.com/deepmind/dsprites-dataset
        if dataset == 'dsprite':
            self.dataset = 'dsprite'
            with np.load(self.data_file, encoding='bytes') as data:
                all_imgs = data['imgs']
                all_imgs = all_imgs[:, :, :, None]  # make into 4d tensor
                all_factors = data['latents_classes']
                all_factors = all_factors[:, 1:]  # Remove color factor
                n_classes = np.array([3, 6, 40, 32, 32])
                self.img_chn=1
                self.img_scale=1.

        elif dataset == '3dshapes':
            self.dataset = '3dshape'
            # imgs = np.load('/home/zl7904/Documents/Data/3dshapes/3dshapes.npy')
            # labels = np.load('/home/zl7904/Documents/Data/3dshapes/3dshapes_label.npy')
            all_imgs, all_factors = self.dataset_reference.get_factorVAE_data()
            # all_imgs = imgs
            # all_factors = labels
            self.class2values=[]
            for i in range(all_factors.shape[1]):
                self.class2values.append(np.unique(all_factors[:,i]))
            n_classes = np.array([10, 10, 10, 8, 4, 15])
            self.img_chn = 3
            self.img_scale = 255.

        # 90% random test/train split
        n_data = all_imgs.shape[0]
        idx_random = np.random.permutation(n_data)
        data_train = all_imgs[idx_random[0: (9 * n_data) // 10]]
        data_test = all_imgs[idx_random[(9 * n_data) // 10:]]
        return data_train, data_test, all_imgs, all_factors, n_classes

    def evaluate_mean_disentanglement(self, model):
        n_tries = 5
        dis_metric = 0
        print("Evaluating disentanglement with " + str(n_tries) + " tries.")
        for i in range(n_tries):
            print("In Try: ", i)
            this_disen_metric = self.evaluate_disentanglement(model)
            print(str(i + 1) + "/" + str(n_tries) + " Disentanglement Metric: " + str(this_disen_metric))
            dis_metric = dis_metric + this_disen_metric
        dis_metric = dis_metric / n_tries

        return dis_metric

    def evaluate_disentanglement(self, model, verbose=False):
        n_examples_per_vote = 100  # Generated examples when we fix a factor (L in paper)
        n_votes = 800  # Total number of training pairs for the classifier
        n_factors = self.n_classes.shape[0]
        n_votes_per_factor = int(n_votes / n_factors)

        # First, we get all the necessary codes at once
        all_mus = []
        all_logvars = []

        code_list = []
        # Fix a factor k
        with torch.no_grad():
            for k_fixed in range(n_factors):
                code_list_per_factor = []
                # Generate training examples for this factor
                for _ in range(n_votes_per_factor):
                    # Fix a value for this factor
                    fixed_value = np.random.choice(self.n_classes[k_fixed])
                    if self.img_chn==3:
                        fixed_value=self.class2values[k_fixed][fixed_value]
                    # Generate data with this factor fixed but all other factors varying randomly. Sample L examples.
                    useful_examples_idx = np.where(self.all_factors[:, k_fixed] == fixed_value)[0]
                    sampled_examples_idx = np.random.choice(useful_examples_idx, n_examples_per_vote)
                    sampled_imgs = self.all_imgs[sampled_examples_idx, :, :, :]
                    # Obtain their representations with the encoder
                    xs = sampled_imgs.reshape([sampled_imgs.shape[0], self.img_chn, 64, 64])/self.img_scale
                    # print("xs.shape: ", xs.shape)
                    xs = torch.from_numpy(xs).float().cuda()
                    code_mu, code_log_sigma = model.encode_for_MIG_new(xs)
                    code_mu = code_mu.cpu().numpy()
                    code_log_sigma = code_log_sigma.cpu().numpy()
                    code_logvar = code_log_sigma * 2
                    all_mus.append(code_mu)
                    all_logvars.append(code_logvar)
                    code_list_per_factor.append((code_mu, code_logvar)) #[160,2,100,6]
                code_list.append(code_list_per_factor) #[5,160,2,100,6]

        # Concatenate every code
        all_mus = np.concatenate(all_mus, axis=0)
        all_logvars = np.concatenate(all_logvars, axis=0)

        # Now, lets compute the KL divergence of each dimension wrt the prior
        emp_mean_kl = self.compute_mean_kl_dim_wise(all_mus, all_logvars)

        # Throw the dimensions that collapsed to the prior
        kl_tol = 1e-2
        useful_dims = np.where(emp_mean_kl > kl_tol)[0]

        # Compute scales for useful dimslabels
        scales = np.std(all_mus[:, useful_dims], axis=0)

        if verbose:
            print("Empirical mean for kl dimension-wise:")
            print(np.reshape(emp_mean_kl, newshape=(-1, 1)))
            print("Useful dimensions:", useful_dims, " - Total:", useful_dims.shape[0])
            print("Empirical Scales:", scales)

        # Dataset for classifier:
        d_values = []
        k_values = []
        # Fix a factor k
        for k_fixed in range(n_factors):
            # Generate training examples for this factor
            for i in range(n_votes_per_factor):
                # Get previously generated codes
                codes = code_list[k_fixed][i][0]
                # Keep only useful dims
                codes = codes[:, useful_dims]
                # Normalise each dimension by its empirical standard deviation over the full data
                # (or a large enough random subset)
                #print("codes : ", codes)
                #print("scales: ", scales)
                norm_codes = codes / scales  # dimension (L, z_dim)
                # Take the empirical variance in each dimension of these normalised representations
                emp_variance = np.var(norm_codes, axis=0)  # dimension (z_dim,), variance for each dimension of code
                # Then the index of the dimension with the lowest variance...
                d_min_var = np.argmin(emp_variance)
                # ...and the target index k provide one training input/output example for the classifier majority vote
                d_values.append(d_min_var)
                k_values.append(k_fixed)
        d_values = np.array(d_values)
        k_values = np.array(k_values)

        # Since both inputs and outputs lie in a discrete space, the optimal classifier is the majority-vote classifier
        # and the metric is the error rate of the classifier (actually they show the accuracy in the paper lol)
        v_matrix = np.zeros((useful_dims.shape[0], n_factors))
        for j in range(useful_dims.shape[0]):
            for k in range(n_factors):
                v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))
        if verbose:
            print("Votes:")
            print(v_matrix)

        # Majority vote is C_j = argmax_k V_jk
        classifier = np.argmax(v_matrix, axis=1)
        predicted_k = classifier[d_values]
        accuracy = np.sum(predicted_k == k_values) / n_votes

        return accuracy

    def compute_mean_kl_dim_wise(self, batch_mu, batch_logvar):
        # Shape of batch_mu is [batch, z_dim], same for batch_logvar
        # KL against N(0,1) is 0.5 * ( var_j - logvar_j + mean^2_j - 1 )
        variance = np.exp(batch_logvar)
        squared_mean = np.square(batch_mu)
        batch_kl = 0.5 * (variance - batch_logvar + squared_mean - 1)
        mean_kl = np.mean(batch_kl, axis=0)
        return mean_kl


def mutual_info_metric_dsprite(vae, dataset_loader, nparams=2, K=None):
    n_samples=10000
    N = len(dataset_loader.dataset)  # number of data samples
    vae.eval()
    q_dist = dist.Normal()
    nparams = q_dist.nparams
    print("nparams: ", nparams)

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    indices = list(range(n_samples))
    batch_size = 128
    total_batch = n_samples // batch_size

    # Loop over all batches
    with torch.no_grad():
        for data in tqdm(dataset_loader):
            xs = data
            batch_size = xs.size(0)
            xs = xs.view(batch_size, 1, 64, 64).cuda()
            mean, log_sigma = vae.encode_for_MIG_new(xs)  # (batch_size, z_dim, 2)
            del xs
            qz_params[n:n + batch_size, :, 0] = mean
            qz_params[n:n + batch_size, :, 1] = log_sigma

            n += batch_size

    qz_params = qz_params
    qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
    qz_samples = q_dist.sample(params=qz_params)

    # --------------------attention: need this block to compute active units and pass it to compute MIG later
    qz_means = qz_params[:, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > 1e-2].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, K))
    # ----------------------------------------------------------------------------------------------------------------

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            q_dist)

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric, metric_axis = compute_metric_with_axis_dsprite(marginal_entropies, cond_entropies, active_units)
    print("MIG Metric: ", metric)
    print("MIG_axis Metric: ", metric_axis)
    print("Marginal_entropies: ", marginal_entropies)
    print("Cond_entropies: ", cond_entropies)

    return metric, marginal_entropies, cond_entropies