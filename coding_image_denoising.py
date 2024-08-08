#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
image_path=("office.png")
image = Image.open(image_path).convert('L') #The image is converted to grayscale
image_array=np.array(image)
print(image_array) #Grayscale values
sigma_values=[0.5,1,2,5,10,50] #Determining Sigma_values to determine the controlling os smoothing
smooth_image=[]

#Gaussian Smoothing
for sigma in sigma_values:
    smooth_image=gaussian_filter(image_array,sigma=sigma)
    smooth_images.append(smooth_images)
    
#plotting the result
plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1), plt.imshow(image_array, cmap='gray'), plt.title('Original')
plt.axis('off')


#Working on Isotropic Linear Diffusion Smoothing

# Load the image
image_path = '/mnt/data/office_noisy.png'
I = img_as_float(io.imread(image_path, as_gray=True))

plt.imshow(I, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.show()
# Parameters
lambda_ = 0.5
t_values = [1, 5, 10, 30, 100]
dt = 0.1  # time step
num_iterations = 1000  # number of iterations for large t

# Initialize the image
u = I.copy()
def gradient(u):
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    ux[:-1, :] = np.diff(u, axis=0)
    uy[:, :-1] = np.diff(u, axis=1)
    return ux, uy

def divergence(ux, uy):
    uxx = np.zeros_like(ux)
    uyy = np.zeros_like(uy)
    uxx[1:, :] = np.diff(ux, axis=0)
    uyy[:, 1:] = np.diff(uy, axis=1)
    return uxx + uyy

def update(u, lambda_, dt):
    ux, uy = gradient(u)
    magnitude = np.sqrt(ux**2 + uy**2)
    D = 1 / (1 + (magnitude / lambda_)**2)
    Dx, Dy = gradient(D * ux), gradient(D * uy)
    div = divergence(Dx[0], Dy[1])
    u += dt * div
    return u

def run_diffusion(I, t_values, lambda_, dt, num_iterations):
    results = []
    u = I.copy()
    for t in range(num_iterations):
        u = update(u, lambda_, dt)
        if t * dt in t_values:
            results.append(u.copy())
    return results

# Run the diffusion process
results = run_diffusion(I, t_values, lambda_, dt, num_iterations)

fig, axes = plt.subplots(1, len(t_values), figsize=(15, 5))
for ax, u, t in zip(axes, results, t_values):
    ax.imshow(u, cmap='gray')
    ax.set_title(f't = {t}')
    ax.axis('off')

plt.show()




#Working on Non-Linear Diffusion

#Non linear diffusion:
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Load the image
image_path = '/mnt/data/office_noisy.png'
I = img_as_float(io.imread(image_path, as_gray=True))

plt.imshow(I, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.show()

# Parameters
lambda_ = 0.5
t_values = [1, 5, 10, 30, 100]
dt = 0.1  # time step
num_iterations = 1000  # number of iterations for large t

# Initialize the image
u = I.copy()

def gradient(u):
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    ux[:-1, :] = np.diff(u, axis=0)
    uy[:, :-1] = np.diff(u, axis=1)
    return ux, uy

def divergence(ux, uy):
    uxx = np.zeros_like(ux)
    uyy = np.zeros_like(uy)
    uxx[1:, :] = np.diff(ux, axis=0)
    uyy[:, 1:] = np.diff(uy, axis=1)
    return uxx + uyy

def update(u, lambda_, dt):
    ux, uy = gradient(u)
    magnitude = np.sqrt(ux**2 + uy**2)
    D = 1 / (1 + (magnitude / lambda_)**2)
    Dx, Dy = gradient(D * ux), gradient(D * uy)
    div = divergence(Dx[0], Dy[1])
    u += dt * div
    return u

def run_diffusion(I, t_values, lambda_, dt, num_iterations):
    results = []
    u = I.copy()
    for t in range(num_iterations):
        u = update(u, lambda_, dt)
        if t * dt in t_values:
            results.append(u.copy())
    return results

# Run the diffusion process
results = run_diffusion(I, t_values, lambda_, dt, num_iterations)


fig, axes = plt.subplots(1, len(t_values), figsize=(15, 5))
for ax, u, t in zip(axes, results, t_values):
    ax.imshow(u, cmap='gray')
    ax.set_title(f't = {t}')
    ax.axis('off')

plt.show()


# Load the noise-free image
image_path_clean = '/mnt/data/office.png'
I_clean = img_as_float(io.imread(image_path_clean, as_gray=True))

plt.imshow(I_clean, cmap='gray')
plt.title('Noise-Free Image')
plt.axis('off')
plt.show()
# Compute the gradient
ux_clean, uy_clean = gradient(I_clean)
magnitude_clean = np.sqrt(ux_clean**2 + uy_clean**2)
# Compute the diffusivity
lambda_ = 0.5
D_clean = 1 / (1 + (magnitude_clean / lambda_)**2)
# Visualize the diffusivity
plt.imshow(D_clean, cmap='gray')
plt.title('Diffusivity Image')
plt.axis('off')
plt.show()


#5_a
lambdas = [0.5, 1, 2, 5, 10]
t_value = 10
dt = 0.1  # time step
num_iterations = int(t_value / dt)  # number of iterations for t = 10
def update(u, lambda_, dt):
    ux, uy = gradient(u)
    magnitude = np.sqrt(ux**2 + uy**2)
    D = 1 / (1 + (magnitude / lambda_)**2)
    Dx, Dy = gradient(D * ux), gradient(D * uy)
    div = divergence(Dx[0], Dy[1])
    u += dt * div
    return u

def run_diffusion(I, lambda_, t_value, dt):
    num_iterations = int(t_value / dt)
    u = I.copy()
    for _ in range(num_iterations):
        u = update(u, lambda_, dt)
    return u

# Run the diffusion process for each lambda
results_lambda = []
for lambda_ in lambdas:
    result = run_diffusion(I, lambda_, t_value, dt)
    results_lambda.append(result)
    
fig, axes = plt.subplots(1, len(lambdas), figsize=(20, 5))
for ax, u, lambda_ in zip(axes, results_lambda, lambdas):
    ax.imshow(u, cmap='gray')
    ax.set_title(f'λ = {lambda_}')
    ax.axis('off')

plt.show()




