import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rigno.dataset import Dataset
from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder
from rigno.stepping import OutputStepper, ResidualStepper, TimeDerivativeStepper
from rigno.stepping import AutoregressiveStepper
from rigno.plot import plot_trajectory, plot_estimates
from rigno.metrics import mse_loss, rel_lp_error_mean

# RANDOM NUMBER GENERATOR SEED
SEED = 45
key = jax.random.PRNGKey(SEED)

# Number of training, validation, and test samples/trajectories
N_TRAIN = 32 * 16
N_VALID = 8 * 8
N_TEST = 16

TAU_MAX_TRAINING = 1
TRAINING_EPOCHS = 50 * 4
BATCH_SIZE = 8
out_dir = '/home/xyh/myCode/rigno-main/output/'

dataset = Dataset(
  datadir='/home/xyh/myData/RIGNO',
  datapath='unstructured/Heat-L-Sines', # Heat-L-Sines
  time_downsample_factor=2,
  space_downsample_factor=1.5,  # per direction
  n_train=N_TRAIN,
  n_valid=N_VALID,
  n_test=N_TEST,
  preload=True,
)
sample = dataset._fetch_mode(idx=[4, 5], mode='valid', get_graphs=False)
fig, axs = plot_trajectory(
  u=sample.u,
  x=sample.x,
  t=sample.t,
  idx_t=[0, 2, 4, 6, 8],
  idx_s=0,
  symmetric=dataset.metadata.signed['u'],
  ylabels=dataset.metadata.names['u'],
  domain=dataset.metadata.domain_x,
)
fig.savefig(out_dir + "trajectory_valid_s0.png", dpi=300, bbox_inches="tight")
plt.close(fig)
fig, axs = plot_trajectory(
  u=sample.u,
  x=sample.x,
  t=sample.t,
  idx_t=[0, 2, 4, 6, 8],
  idx_s=1,
  symmetric=dataset.metadata.signed['u'],
  ylabels=dataset.metadata.names['u'],
  domain=dataset.metadata.domain_x,
)
fig.savefig(out_dir + "trajectory_valid_s1.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figures to: {out_dir}")

# Feel free to change the number of parameters as you wish
model = RIGNO(
  num_outputs=dataset.shape[-1],
  processor_steps=6 + 2,
  node_latent_size=32 * 2,
  edge_latent_size=32 * 2,
  mlp_hidden_layers=1 + 2,
  concatenate_tau=(True if dataset.time_dependent else False),
  concatenate_t=(True if dataset.time_dependent else False),
  conditioned_normalization=(True if dataset.time_dependent else False),
  cond_norm_hidden_size=16,
  p_edge_masking=0.5,
)

# Let's use the time-derivative stepper
# You can also use OutputStepper or ResidualStepper
stepper = TimeDerivativeStepper(operator=model)

# AutoregressiveStepper is a wrapper class for all time stepping strategies
autoregressive = AutoregressiveStepper(stepper=stepper, dt=dataset.dt)

# We set our graph construction settings
graph_builder = RegionInteractionGraphBuilder(
  periodic=dataset.metadata.periodic,
  rmesh_levels=4,
  subsample_factor=4,
  overlap_factor_p2r=1.0,
  overlap_factor_r2p=2.0,
  node_coordinate_freqs=4,
)

# We use the graph builder defined above to create graph metadata
dataset.build_graphs(builder=graph_builder)

# We comptue the relevant statistics of the dataset
dataset.compute_stats(residual_steps=TAU_MAX_TRAINING)
stats = {
  key: {
    k: (jnp.array(v) if (v is not None) else None)
    for k, v in val.items()
  }
  for key, val in dataset.stats.items()
}


# Create dummy inputs for model initialization
dummy_graph_builder = RegionInteractionGraphBuilder(
  periodic=dataset.metadata.periodic,
  rmesh_levels=1,
  subsample_factor=4,
  overlap_factor_p2r=.01,
  overlap_factor_r2p=.01,
  node_coordinate_freqs=4,
)
dummy_graphs = dummy_graph_builder.build_graphs(
  dummy_graph_builder.build_metadata(
  x_inp=dataset.sample.x[0, 0],
  x_out=dataset.sample.x[0, 0],
  domain=np.array(dataset.metadata.domain_x),
))
dummy_graphs = jax.tree_util.tree_map(lambda v: jnp.repeat(v, repeats=BATCH_SIZE, axis=0), dummy_graphs)
dummy_inputs = Inputs(
  u=jnp.ones(shape=(BATCH_SIZE, 1, *dataset.sample.u.shape[2:])),
  c=None,
  x_inp=dataset.sample.x,
  x_out=dataset.sample.x,
  t=jnp.zeros(shape=(BATCH_SIZE, 1)),
  tau=jnp.ones(shape=(BATCH_SIZE, 1)),
)

# Initialize the model
subkey, key = jax.random.split(key)
variables = model.init(subkey, inputs=dummy_inputs, graphs=dummy_graphs)

# Report number of parameters
n_model_parameters = np.sum(
  jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.prod(x.shape).item(), variables['params']))[0]
).item()
print(f'We are going to train a RIGNO with (only) {n_model_parameters/1000:.1f}K parameters.')

# Set the learning rate
UPDATES_PER_EPOCH = (dataset.shape[1] - 1) * (N_TRAIN / BATCH_SIZE)
lr = optax.exponential_decay(
  transition_steps=(UPDATES_PER_EPOCH*TRAINING_EPOCHS),
  init_value=1e-02,
  decay_rate=.1,
)
tx = optax.chain(optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-08))
# Create a training state
state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

@jax.jit
def _compute_loss(params, u_inp, x_inp, t_inp, tau, u_tgt, x_out, key):
  """Computes the prediction of the model and returns its loss."""

  # Get the output
  key, subkey = jax.random.split(key)
  inputs = Inputs(
    u=u_inp,
    c=None,
    x_inp=x_inp,
    x_out=x_out,
    t=t_inp,
    tau=tau,
  )
  _loss_inputs = stepper.get_loss_inputs(
    variables={'params': params},
    stats=stats,
    u_tgt=u_tgt,
    inputs=inputs,
    graphs=graph_builder.build_graphs(batch.g),
    key=subkey,
  )

  return mse_loss(*_loss_inputs)

# Let's train a RIGNO!
init_times = jnp.arange(dataset.shape[1] - 1)
best = {'params': state.params, 'loss': 1e+08}
for epoch in range(TRAINING_EPOCHS):
  loss_epoch = 0

  # # NOTE: We can randomly re-build the graph so that the training is
  # # done every time with a new set of regional nodes.
  # # This step is necessary for achieving resolution invariance
  dataset.build_graphs(builder=graph_builder, key=subkey)

  # Get the batches with shuffling
  subkey, key = jax.random.split(key)
  batches = dataset.batches(mode='train', batch_size=BATCH_SIZE, key=subkey)

  for i_batch, batch in enumerate(batches):
    # Get loss and updated state
    subkey, key = jax.random.split(key)

    # Pair input-outputs from the trajectories (subbatches)
    u_inp_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=batch.u,
        start_index=(lt), slice_size=1, axis=1)
    )(init_times)
    x_inp_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=batch.x,
        start_index=(lt), slice_size=1, axis=1)
    )(init_times)
    t_inp_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=batch.t,
        start_index=(lt), slice_size=1, axis=1)
    )(init_times)
    u_tgt_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=jnp.concatenate([batch.u, jnp.zeros_like(batch.u)], axis=1),
        start_index=(lt+1), slice_size=TAU_MAX_TRAINING, axis=1)
    )(init_times)
    t_tgt_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=jnp.concatenate([batch.t, jnp.zeros_like(batch.t)], axis=1),
        start_index=(lt+1), slice_size=TAU_MAX_TRAINING, axis=1)
    )(init_times)
    x_out_batch = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=jnp.concatenate([batch.x, jnp.zeros_like(batch.x)], axis=1),
        start_index=(lt+1), slice_size=TAU_MAX_TRAINING, axis=1)
    )(init_times)
    # Repeat inputs along the time axis to match with u_tgt
    # -> [init_times, BATCH_SIZE, TAU_MAX_TRAINING, ...]
    u_inp_batch = jnp.tile(u_inp_batch, reps=(1, 1, TAU_MAX_TRAINING, 1, 1))
    x_inp_batch = jnp.tile(x_inp_batch, reps=(1, 1, TAU_MAX_TRAINING, 1, 1))
    t_inp_batch = jnp.tile(t_inp_batch, reps=(1, 1, TAU_MAX_TRAINING, 1, 1))
    # Get the lead times
    tau_batch = t_tgt_batch - t_inp_batch

    # Compute and apply gradients for each subbatch
    for i_subbatch in range(len(init_times)):
      subkey, key = jax.random.split(key)
      loss, grads = jax.value_and_grad(_compute_loss)(
        state.params, u_inp_batch[i_subbatch], x_inp_batch[i_subbatch], t_inp_batch[i_subbatch],
        tau_batch[i_subbatch], u_tgt_batch[i_subbatch], x_out_batch[i_subbatch], key=subkey
      )
      # Apply gradients
      state = state.apply_gradients(grads=grads)
      # Add to the epoch loss
      loss_epoch += loss / UPDATES_PER_EPOCH

  # STORE PARAMETERS
  _lr = state.opt_state[-1].hyperparams['learning_rate'].item()
  if loss_epoch < best['loss']:
    best = {'params': state.params, 'loss': loss_epoch}
  # PRINT LOSS
  print(f'EPOCH {epoch+1:04d}/{TRAINING_EPOCHS} \t LR {_lr:.2e} \t LOSS {loss:.2e}')

batch = next(dataset.batches(mode='test', batch_size=N_TEST))

u_model, _ = autoregressive.unroll(
  variables={'params': best['params']},  # Our trained parameters
  stats=stats,  # Dataset statistics
  num_steps=10,  # We want a trajectory of 10 time steps just like the ground-truth trajectory
  inputs=Inputs(
    u=batch.u[:, :1],  # Initial condition
    c=None,  # No known spatial parameters
    x_inp=batch.x[:, :1],  # Coordinates
    x_out=batch.x[:, :1],  # Coordinates, again
    t=batch.t[:, :1],  # Initial time
    tau=None,  # AutoregressiveStepper.unroll is based on a fixed dt, which is given to it beforehand
  ),
  graphs=graph_builder.build_graphs(batch.g),  # Build the actual graphs from the light-weight metadata
)

assert np.allclose(u_model[:, 0], batch.u[:, 0])

# Let's check the predictions for the last time step
target_time_index = -1

# Median relative L1 test error
rel_l1_error = np.median(rel_lp_error_mean(batch.u[:, [target_time_index]], u_model[:, [target_time_index]], p=1))
print(f'Median relative L1 test error over all test samples: {rel_l1_error * 100:.2f}%')
if (rel_l1_error * 100) < 10:
  print(f'Not bad for only {n_model_parameters/1000:.1f}K parameters and {N_TRAIN} training trajectories!')


# Change the index to see more samples
sample_index = 3

fig = plot_estimates(
  u_inp=batch.u[sample_index, 0],
  u_gtr=batch.u[sample_index, target_time_index],
  u_prd=u_model[sample_index, target_time_index],
  x_inp=batch.x[sample_index, target_time_index],
  x_out=batch.x[sample_index, target_time_index],
  symmetric=dataset.metadata.signed['u'],
  names=dataset.metadata.names['u'],
  domain=dataset.metadata.domain_x,
)
fig.savefig(out_dir + "trajectory_valid_s3.png", dpi=300, bbox_inches="tight")
plt.close(fig)

rel_l1_error = np.median(rel_lp_error_mean(batch.u[[sample_index], [target_time_index]], u_model[[sample_index], [target_time_index]], p=1))
print(f'Relative L1 test error of SAMPLE #{sample_index:02d}: {rel_l1_error * 100:.2f}%')