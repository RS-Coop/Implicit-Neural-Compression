import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize, LogNorm
import gif
import warnings

'''
Returns a method for plotting a set of features.

NOTE: In the future, this could use the adjacency
'''
def get_plot_func(points):

    plot_func = None

    if points.shape[1] == 2:
        #triangulate and save
        triangulation = Triangulation(points[:,0], points[:,1])

        #make plot function
        def plot_func(f, ax, norm=Normalize, vmin=None, vmax=None, **kwargs):
            return ax.tripcolor(triangulation, f, norm=norm(vmin=vmin, vmax=vmax), **kwargs)

    else:
        warnings.warn("Mesh plotting only supported for 2D data.")

    return plot_func

'''
Generates a side-by-side GIF of the raw data and the model reconstruction for the
test dataset; logs the result.

Input:
    trainer: lightning trainer
    datamodule: data module
    model: model to use or none to use best checkpoint
'''
def make_gif(trainer, datamodule, model):
    #run on test data
    #NOTE: This context shouldn't be required, but something must be broken in Lightning
    with torch.inference_mode():
        data = trainer.predict(model=model, datamodule=datamodule)

    data = datamodule.predict.denormalize(torch.stack(data))

    #get original data
    raw = datamodule.predict.getall()

    #build residual
    res = torch.abs(raw - data)

    #data limits
    vmin, vmax = torch.amin(raw, dim=(0,1)), torch.amax(raw, dim=(0,1))
    rmin, rmax = torch.amin(res, dim=(0,1))+1e-8, torch.amax(res, dim=(0,1))

    #get plotting function
    plot_func = get_plot_func(datamodule.predict.p)

    if plot_func == None:
        return

    #gif frame closure
    @gif.frame
    def plot(i, c):
        fig, axis = plt.subplots(3, 1, figsize=(8,12), constrained_layout=True)

        im1 = plot_func(raw[i,:,c], axis[0], vmin=vmin[c], vmax=vmax[c])
        axis[0].set_title("Uncompressed")

        im2 = plot_func(data[i,:,c], axis[1], vmin=vmin[c], vmax=vmax[c])
        axis[1].set_title("Reconstructed")

        im3 = plot_func(res[i,:,c], axis[2], norm=LogNorm, vmin=rmin[c], vmax=rmax[c])
        axis[2].set_title("Residual")

        for ax in axis:
            ax.axis('off')
            ax.set_aspect("auto", "box")

        # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(im1, ax=axis[0], location='left')
        fig.colorbar(im2, ax=axis[1], location='left')
        fig.colorbar(im3, ax=axis[2], location='left')

    for c in range(data.shape[2]):
        print(f"\nMaking GIF for channel {c}...")

        #build frames
        frames = [plot(i, c) for i in range(data.shape[0])]

        #tight bounding box
        # gif.options.matplotlib["bbox"] = 'tight'

        #save gif
        gif.save(frames, f'{trainer.logger.log_dir}/predict_channel-{c}.gif', duration=100)

    print("GIFs completed")

    return
