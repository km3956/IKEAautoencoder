import sys
import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model
from ae1 import AE1
from ae2 import AE2


class AE3(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 16, 16),
            layers=[
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ]
        )

        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 8, 8),
            layers=[
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ]
        )

        self.model = Model(
            input_shape=self.encoder.input_shape,
            layers=[
                self.encoder,
                self.decoder
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load('data', image_size=64)
    data.shuffle()

    ae1 = AE1('models/ae1.pt')
    ae1.load()
    encoded_data_ae1 = ae1.encode(data)

    ae2 = AE2('models/ae2.pt')
    ae2.load()
    encoded_data_ae2 = ae2.encode(encoded_data_ae1)

    ae3 = AE3('models/ae3.pt')
    ae3.print()

    if not epochs:
        print(f'\nLoading {ae3.path}...')
        ae3.load()
    else:
        train_data, test_data = encoded_data_ae2.split(0.8)
        print(f'\nTraining...')
        ae3.train(epochs, train_data)
        print(f'\nSaving {ae3.path}...')
        ae3.save()

    print(f'\nGenerating samples...')
    samples = ae3.generate(encoded_data_ae2)
    data.display(32)
    samples.display(32)
