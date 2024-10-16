# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--vel_pursuer', type=float, default=0.2, required=False, help='Velocity of Pursuer')
p.add_argument('--vel_evader', type=float, default=0.2, required=False, help='Velocity of Evader')
p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='none', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.PursuitEvasionOneVOne(numpoints=65000, collisionR=opt.collisionR, vel_pursuer=opt.vel_pursuer, 
                                          vel_evader=opt.vel_evader, pretrain=opt.pretrain, tMin=opt.tMin,
                                          tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                          pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                          num_src_samples=opt.num_src_samples) # EDIT THIS


dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_PE_1v1(dataset, opt.minWith)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Validation Function
def val_fn(model, ckpt_dir, epoch):
  return 
  # # Time values at which the function needs to be plotted
  # times = [0., 0.5*(opt.tMax - 0.1), (opt.tMax - 0.1)]
  # num_times = len(times)

  # # # Theta slices to be plotted
  # # thetas = [-math.pi, -0.5*math.pi, 0., 0.5*math.pi, math.pi]
  # # num_thetas = len(thetas)

  # # Create a figure
  # fig = plt.figure(figsize=(5*num_times, 5*num_times)) # 5*num_thetas

  # # Get the meshgrid in the (x, y) coordinate
  # sidelen = 50
  # mgrid_coords = dataio.get_mgrid(sidelen,4)

  # # Start plotting the results
  # for i in range(num_times):
  #   time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

  #   coords = torch.cat((time_coords, mgrid_coords), dim=1) 
  #   num_batches = 100
  #   batch_size = coords.shape[0] // num_batches
  #   model_out = []
  #   for i in range(num_batches):  # Dividing into 100 batches
  #     start_index = i * batch_size
  #     end_index = start_index + batch_size if i < num_batches - 1 else coords.shape[0]  # Ensure the last batch includes remaining elements
  #     sub_coords = coords[start_index:end_index, :]
  #     model_in = {'coords': sub_coords.cuda()}
  #     model_sub_out = model(model_in)['model_out']
  #     model_out.append(model_sub_out.cpu())

  #     torch.cuda.synchronize()
  #     del sub_coords, model_sub_out
  #     torch.cuda.empty_cache()

  #     # Print CUDA memory usage
  #     print(f"After iteration {i + 1}:")
  #     print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
  #     print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


  #   # Detatch model ouput and reshape
  #   model_out = torch.cat(model_out, dim=0)
  #   model_out = model_out.reshape((sidelen, sidelen, sidelen, sidelen))

  #   # Unnormalize the value function
  #   norm_to = 0.02
  #   mean = 0.25
  #   var = 0.5
  #   model_out = (model_out*var/norm_to) + mean 

  #   # Plot the zero level sets
  #   model_out = (model_out <= 0.001)*1.

  #   # Plot the actual data
  #   s = fig.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
  #   fig.colorbar(s) 

  # fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))
  

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload)

