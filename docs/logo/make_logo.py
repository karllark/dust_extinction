import argparse
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from dust_extinction.parameter_averages import G23

def rect(x,y,w,h,c):
    ax = plt.gca()
    polygon = plt.Rectangle((x,y),w,h,color=c)
    ax.add_patch(polygon)

def rainbow_fill(X,Y, cmap=plt.get_cmap("jet")):
    plt.plot(X,Y,lw=0)  # Plot so the axes scale correctly

    dx = X[1]-X[0]
    N  = float(X.size)

    for n, (x,y) in enumerate(zip(X,Y)):
        color = cmap(n/N)
        rect(x,0,dx,y,color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ext = G23()

    lam = np.logspace(np.log10(0.0912), np.log10(30.0), num=1000) * u.micron

    ext1 = ext(lam)
    ext2 = np.log10(ext1)

    weight = np.log10(lam.value) / (np.log10(30.0) - np.log10(0.0912))

    #ax.plot(lam, ext1, "k-", lw=2)
    #ax.plot(lam, ext2 + 1.5, "b-", lw=2)
    mergeext = ext1 + (weight * (ext2 + 0.5)) + 1.0
    # im = ax.plot(lam, mergeext, "g-", lw=2)

    #patch = patches.Circle((1.0, 3.0), radius=3, transform=ax.transData)
    #im[0].set_clip_path(patch)

    rainbow_fill(lam.value, mergeext)

    ax.axis('off')

    ax.set_xscale("log")

    # show the figure or save it to a pdf file
    if args.png:
        fig.savefig("dust_extinction_log.png")
        plt.close()
    else:
        plt.show()