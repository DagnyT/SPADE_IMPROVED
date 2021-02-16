import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:

    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def add_scalars_to_tensorboard(self, type, epoch, iter, loss_G, loss_D):

        for name, value in loss_G.items():
            self.writer.add_scalar(type+'/'+name, value.mean().item(), iter)

        for name, value in loss_D.items():
            self.writer.add_scalar(type+'/'+name, value.mean().item(), iter)

    def add_images_to_tensorboard(self, img, name):
        self.writer.add_image('images/'+name, img, 0)

class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = os.path.join(path, 'log.txt')
        self.init_logs()

    def init_logs(self):

        text_file = open(self.path, "w")
        text_file.close()

    def log_string(self, string):
        """Stores log string in log file."""
        text_file = open(self.path, "a")
        text_file.write(str(string)+'\n')
        text_file.close()