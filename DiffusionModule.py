import torch


class Diffusion:
    
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02 ):
        

        
        
        self.timesteps = timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        
        self.alphas = 1.0 - self.betas 
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat((torch.tensor([1.0], dtype=torch.float32), self.alpha_cumprod[:-1]))
        
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod) 
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)
        self.posterior_variance = self.betas * (1. - self.alpha_cumprod_prev) / (1. - self.alpha_cumprod)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alpha_cumprod_prev) / (1. - self.alpha_cumprod))
        self.posterior_mean_coef2 = ((1. - self.alpha_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alpha_cumprod))
        

    @staticmethod
    def pick_t_index_expand_to_shape(a: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        
        batch_size = t.shape[0]
        
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(target_shape) - 1)))


    def q_xt_given_x0(self, x_0, t , noise=None):
        
        
        
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.pick_t_index_expand_to_shape(self.sqrt_alpha_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self.pick_t_index_expand_to_shape(self.sqrt_one_minus_alpha_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise 
        
        return x_t, noise
    

    
    @torch.no_grad()
    def p_xtminus1_given_xt(self, x_t, t, model, clip_denoised=True, repeat_noise=False):
        
        
        posterior_variance_t = self.pick_t_index_expand_to_shape(self.posterior_variance, t, x_t.shape)
        posterior_mean_coef1_t = self.pick_t_index_expand_to_shape(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self.pick_t_index_expand_to_shape(self.posterior_mean_coef2, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self.pick_t_index_expand_to_shape(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape)
        
        sqrt_recip_alpha_cumprod = 1.0 / self.sqrt_alpha_cumprod
        sqrt_recip_alpha_cumprod_t = self.pick_t_index_expand_to_shape(sqrt_recip_alpha_cumprod, t, x_t.shape)

        predicted_noise = self.predict_noise(x_t, t, model)
        x_0_predicted = sqrt_recip_alpha_cumprod_t * (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise)

        if clip_denoised:
            x_0_predicted.clamp_(-1., 1.)

        posterior_mean = posterior_mean_coef1_t * x_0_predicted + posterior_mean_coef2_t * x_t

        if t[0] == 0:
            return posterior_mean
        else:
            if not repeat_noise:
                
                noise = torch.randn_like(x_t)
            else:
                
                noise = torch.randn((1, *x_t.shape[1:]), dtype=x_t.dtype, device=x_t.device)
            
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise 
            
            
    @torch.no_grad()
    def sample(self, shape, model, device, repeat_noise=False, x_start=None):

        if x_start is None:
            
            img = torch.randn(shape, device=device)
        else:
            img = x_start.to(device)

        for i in reversed(range(0, self.timesteps)):
            
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_xtminus1_given_xt(img, t, model, repeat_noise=repeat_noise)

        return img


