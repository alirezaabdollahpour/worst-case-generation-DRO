
Reproduction scaffold for:
"Worst-case generation via minimax optimization in Wasserstein space"
(CIFAR-10 Figure 4-style interpolation panel, plus the latent-space minimax training loop).

What is implemented (per the paper):
- Convolutional VAE latent space for CIFAR-10 (Appendix B.4, Table A.2)
- Latent-space MLP classifier θ0 (Appendix B.2, Table A.2)
- Label-conditioned residual transport map T_φ with identity initialization (Appendix B.2, Table A.2)
- Particle gradient descent-ascent (GDA) with momentum + matching loss training of T_φ (Algorithm 1, Appendix B.2)
- Visualization script that linearly interpolates between z and T_φ(z) in latent space and decodes via the VAE (Figure 4(b)-style)

Important note:
- The paper does not explicitly state the CIFAR-10 value of γ (quadratic penalty strength) in the figure 4 paragraph.
  The minimax training script exposes --gamma as a flag (default: 8.0).

Install:
    pip install -r requirements.txt

Train VAE (Table A.2 defaults):
    python train_vae_cifar10.py --data_root ./data --out_dir ./runs/cifar10_vae

(Optional) Track VAE training with Weights & Biases:
    python train_vae_cifar10.py --data_root ./data --out_dir ./runs/cifar10_vae --wandb \
        --wandb_project worst-case-generation-MLP --wandb_log_images_every 10

Train classifier θ0 (Table A.2 defaults):
    python train_classifier_cifar10.py --data_root ./data \
        --vae_ckpt ./runs/cifar10_vae/vae_final.pt \
        --out_dir ./runs/cifar10_classifier

Run minimax training (Figure 4 CIFAR-10 settings, Table A.2 transport optimizer):
    python train_minimax_cifar10.py --data_root ./data \
        --vae_ckpt ./runs/cifar10_vae/vae_final.pt \
        --clf_ckpt ./runs/cifar10_classifier/classifier_final.pt \
        --out_dir ./runs/cifar10_minimax

Visualize Figure 4(b)-style panel:
    python visualize_fig4_cifar10.py --data_root ./data \
        --vae_ckpt ./runs/cifar10_vae/vae_final.pt \
        --minimax_ckpt ./runs/cifar10_minimax/minimax_final.pt \
        --out_path ./runs/fig4_cifar10.png
