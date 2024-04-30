import torch

import matplotlib.pyplot as plt
from core.modules.metrics import r3error

'''
Generates a side-by-side GIF of the raw data and the model reconstruction for the
test dataset; logs the result.

Input:
    trainer: lightning trainer
    datamodule: data module
    model: model to use or none to use best checkpoint
'''
def make_error_plots(trainer, datamodule, model):
    #run on test data
    #NOTE: This context shouldn't be required, but something must be broken in Lightning
    with torch.inference_mode():
        data = trainer.predict(model=model, datamodule=datamodule)

    data = torch.cat(data).reshape(len(datamodule.predict), -1, datamodule.output_shape[1])

    data = datamodule.predict.denorm_f(data)

    #get original data
    raw = datamodule.predict.get_features()

    #build residual
    error = r3error(data, raw)

    #plot
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    for c in range(error.shape[1]):
        ax.semilogy(error[:,c])

    ax.set_title("Snapshot R3Error")
    ax.legend([f'Channel {c}' for c in range(error.shape[1])])

    #save
    plt.savefig(f'{trainer.logger.log_dir}/r3error.pdf')