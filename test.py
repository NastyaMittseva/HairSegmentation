from network.solver import HairSegmentation
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0'

training_data_path = 'CelebAMask-HQ-hair/train_label/'
valid_data_path = 'CelebAMask-HQ-hair/val_label/'
test_data_path = 'CelebAMask-HQ-hair/test_label/'
resolution = 224
num_classes = 2

decay_epoch = 100
lr = 1
rho = 0.95
eps = 1e-7
decay = 2e-5
gradient_loss_weight = 0.125
        
resume_epochs = 99
log_step = 100
sample_step = 300
num_epochs = 99
batch_size = 64

train_results_dir = './experiments/start2/train_results/'
valid_results_dir = './experiments/start2/valid_results/'
test_results_dir = './experiments/start2/test_results/'
model_save_dir = './experiments/start2/models/'
log_dir = model_save_dir + 'logs/'

model = HairSegmentation(training_data_path, valid_data_path, test_data_path, resolution, num_classes, decay_epoch, lr, rho, eps, decay, 
                         gradient_loss_weight, resume_epochs, log_step, sample_step, num_epochs, batch_size, train_results_dir, 
                         valid_results_dir, test_results_dir, model_save_dir, log_dir)
model.test()