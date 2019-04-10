from GP_DIP_utils import *
import os

for id in range(50):
    print(id)
    path = 'SAMPLES/%d.npy' % id
    if not os.path.exists(path):
        continue
    if id == 0:
        samples = load_objs(path)
    else:
        samples = np.vstack((samples, load_objs(path)))

# non-stationary
# kernel = compute_kernel(samples)
# save_objs('kernel_2layer_skip_800K.npy', kernel)

# stationary
kernel = compute_mean_cov_from_samples(samples)
save_objs('stationary_kernel_2layer_skip_800K.npy', kernel)
