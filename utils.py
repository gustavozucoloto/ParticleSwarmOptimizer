import numpy as np
import matplotlib.pyplot as plt

def plot_3D_function(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_contour_function(X, Y, Z):
    x_min = X.ravel()[X.argmin()]
    y_min = Y.ravel()[Y.argmin()]
    x_max = X.ravel()[X.argmax()]
    y_max = Y.ravel()[Y.argmax()]
    
    plt.figure(figsize=(8,6))
    plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = plt.contour(X, Y, Z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    plt.show()


def plot_state(X, Y, Z):
    fig = plt.figure(figsize=(14, 6))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Contour plot
    x_min = X.ravel()[X.argmin()]
    y_min = Y.ravel()[Y.argmin()]
    x_max = X.ravel()[X.argmax()]
    y_max = Y.ravel()[Y.argmax()]
    ax2 = fig.add_subplot(122)
    x_min = X.ravel()[Y.argmin()]
    y_min = Y.ravel()[Y.argmin()]
    contour = ax2.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = ax2.contour(X, Y, Z, 5, colors='black', alpha=0.4)
    ax2.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
