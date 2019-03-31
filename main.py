from keras.callbacks import ModelCheckpoint
from model import *
from data import *
import numpy as np
import skimage.io as io

# batch 2 steps 5 epochs 3 - игловидные
# batch 3 steps 5 epochs 2 - дендриты
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect')

test_num = 13
#
# myGene = trainGenerator(6, 'data/train_class2', 'image', 'label', data_gen_args, save_to_dir="data/train_class2/aug")
#
# model = unet(pretrained_weights="unet_membrane_2class.hdf5")
# model_checkpoint = ModelCheckpoint('unet_membrane_2class.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit_generator(myGene, steps_per_epoch=5, epochs=1, callbacks=[model_checkpoint])

# testGene = testGenerator("data/test_class2", num_image=test_num)
# results = model.predict_generator(testGene, test_num, verbose=1)
# saveResult("data/test_class2", results)

# -------------------------------------------------------------------------------------------------
mix_num = 8

data = []
for i in range(0, mix_num):
    data.append(io.imread(os.path.join("data\objects_mask\mixed\%d.png" % i), as_gray=True))


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1
    )
    count = 0
    for x_index in range(S.shape[0]):
        for y_index in range(S.shape[1]):
            if S[x_index][y_index] > 0 & S[x_index][y_index] < k * k:
                count += 1
    return count

for i in range(0, mix_num):

    # Minimal dimension of image
    p = min(data[i].shape)
    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))
    # Extract the exponent
    n = int(np.log(n) / np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(data[i], size))
        # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    print(-coeffs[0])


