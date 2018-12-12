from tqdm import tqdm
from options.train_options import TrainOptions
from datasets import CreateDataLoader
from models import create_model
from utils.visualizer import Visualizer
from utils import util

def get_state_message(epoch, n_iter, param_dict):
    log = "Epoch: {}, n_iter: {}\n".format(epoch, n_iter)
    for name, value in param_dict.items():
        log += "{}: {:.5f}, ".format(name, value)

    return log


def train():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    model = create_model(opt)
    model.setup(opt)
    util.mkdir(model.save_dir)
    visualizer = Visualizer(opt)

    n_iter = (opt.epoch_count - 1) * dataset_size
    total_epoch = opt.niter + opt.niter_decay + 1

    for epoch in range(opt.epoch_count, total_epoch):
        model.update_epoch_params(epoch)
        for data in tqdm(dataset, total=dataset_size, ascii=True):
            model.set_input(data)
            model.optimize_parameters()

            n_iter += 1
            if n_iter % opt.display_freq == 0:
                for name, image in model.get_current_visuals().items():
                    visualizer.add_image(name, image, n_iter)

            if n_iter % opt.print_freq == 0:
                for name, value in model.get_current_losses().items():
                    visualizer.add_scalar(name, value, n_iter)
                log = get_state_message(epoch, n_iter, model.get_current_losses())
                tqdm.write(log)

            if n_iter % opt.save_lastest_freq == 0:
                model.save_networks('latest')


        log = get_state_message(epoch, n_iter, model.get_current_losses())
        visualizer.add_log(log)

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        model.update_learning_rate()


if __name__ == '__main__':
    train()
