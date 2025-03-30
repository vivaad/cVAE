import torch 
import matplotlib.pyplot as plt
from cpu_cvae import CVAE, device, latent_size

model = CVAE(28*28, latent_size, 10).to(device)
model.load_state_dict(torch.load('cvae_model.pth'))
model.eval()

with torch.no_grad():
    class_index = 0
    c = torch.zeros(10, 10).to(device)
    c[:, class_index] = 1
    sample = torch.randn(10, latent_size).to(device)
    sample = model.decode(sample, c).cpu()
    # save_image(sample.view(10, 1, 28, 28), 'sample_class_10.png')

    # # Print the sample tensor as an image
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(sample[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()
