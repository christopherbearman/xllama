// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LCMScheduler — a header-only, dependency-free reimplementation of the diffusers
// LCMScheduler as configured by LCM_Dreamshaper_v7 (scaled_linear betas,
// epsilon prediction, timestep_spacing="leading", original_inference_steps=50,
// set_alpha_to_one=true, timestep_scaling=10). Pure double math so it builds and
// is checked on the host against the Python reference before it ships in the
// console pipeline (uwp/diffuse.cpp).
//
// Reference (diffusers/schedulers/scheduling_lcm.py, pipeline_latent_consistency):
//   betas  = (linspace(sqrt(b0), sqrt(b1), N))^2 ; acp = cumprod(1 - betas)
//   set_timesteps(n):
//     k = N / original_steps                       (1000 / 50 = 20)
//     origin = [i*k - 1 for i in 1..original_steps] (19, 39, ..., 999)
//     idx = floor(linspace(0, original_steps, n, endpoint=False))
//     timesteps = reversed(origin)[idx]             (n=4: 999, 759, 499, 259)
//   init_noise_sigma = 1 ; scale_model_input = identity
//   step(eps, t, x), prev_t = next timestep (or t on the last step):
//     a_t = acp[t] ; a_prev = acp[prev_t] (last step: 1.0, set_alpha_to_one)
//     scaled = t * timestep_scaling
//     c_skip = 0.25 / (scaled^2 + 0.25) ; c_out = scaled / sqrt(scaled^2 + 0.25)
//     x0 = (x - sqrt(1 - a_t) * eps) / sqrt(a_t)
//     denoised = c_out * x0 + c_skip * x
//     not last: x' = sqrt(a_prev) * denoised + sqrt(1 - a_prev) * noise
//     last:     x' = denoised
//   guidance embedding (w = guidance_scale - 1, dim 256):
//     w *= 1000 ; half = dim/2 ; f = exp(-log(10000) * j / (half - 1))
//     emb = [sin(w*f_j) for j] ++ [cos(w*f_j) for j]
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace xllama::diffusion {

struct LcmConfig {
    int num_train_timesteps = 1000;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    int original_inference_steps = 50;
    double timestep_scaling = 10.0;
};

class LcmScheduler {
  public:
    explicit LcmScheduler(const LcmConfig& cfg = {}) : cfg_(cfg) {
        const int N = cfg_.num_train_timesteps;
        const double sb0 = std::sqrt(cfg_.beta_start);
        const double sb1 = std::sqrt(cfg_.beta_end);
        double acp = 1.0;
        acp_.resize(N);
        for (int i = 0; i < N; ++i) {
            const double s = sb0 + (sb1 - sb0) * (double)i / (double)(N - 1);
            acp *= (1.0 - s * s);
            acp_[i] = acp;
        }
    }

    // Configure the inference schedule (LCM: 2-8 steps; 4 recommended).
    void set_timesteps(int n) {
        const int N = cfg_.num_train_timesteps;
        const int orig = cfg_.original_inference_steps;
        if (n < 1)
            n = 1;
        if (n > orig)
            n = orig;
        const int k = N / orig;
        std::vector<double> origin_rev; // descending: 999, 979, ..., 19
        for (int i = orig; i >= 1; --i)
            origin_rev.push_back((double)(i * k - 1));
        timesteps_.clear();
        for (int j = 0; j < n; ++j) {
            const int idx = (int)std::floor((double)j * (double)orig / (double)n);
            timesteps_.push_back(origin_rev[idx]);
        }
        step_index_ = 0;
    }

    const std::vector<double>& timesteps() const {
        return timesteps_;
    }
    double init_noise_sigma() const {
        return 1.0;
    }
    int step_index() const {
        return step_index_;
    }
    bool is_last_step() const {
        return step_index_ + 1 >= (int)timesteps_.size();
    }
    // LCM does not scale the model input.
    template <typename T> void scale_model_input(std::vector<T>&) const {}

    // One LCM step. eps = UNet epsilon prediction, x = current latent (updated in
    // place), noise = fresh N(0,1) of equal length (ignored on the last step).
    template <typename T>
    void step(const std::vector<T>& eps, std::vector<T>& x, const std::vector<T>& noise) {
        const int t = (int)timesteps_[step_index_];
        const bool last = is_last_step();
        const double a_t = acp_[t];
        const double a_prev = last ? 1.0 : acp_[(int)timesteps_[step_index_ + 1]];
        const double scaled = (double)t * cfg_.timestep_scaling;
        const double sd2 = 0.25; // sigma_data = 0.5
        const double c_skip = sd2 / (scaled * scaled + sd2);
        const double c_out = scaled / std::sqrt(scaled * scaled + sd2);
        const double sa = std::sqrt(a_t), sb = std::sqrt(1.0 - a_t);
        const double sp = std::sqrt(a_prev), sq = std::sqrt(1.0 - a_prev);
        for (size_t i = 0; i < x.size(); ++i) {
            const double xi = (double)x[i];
            const double x0 = (xi - sb * (double)eps[i]) / sa;
            const double den = c_out * x0 + c_skip * xi;
            x[i] = last ? (T)den : (T)(sp * den + sq * (double)noise[i]);
        }
        ++step_index_;
    }

    // timestep_cond input for the LCM UNet: w = guidance_scale - 1.
    static std::vector<float> guidance_embedding(double guidance_scale, int dim = 256) {
        const double w = (guidance_scale - 1.0) * 1000.0;
        const int half = dim / 2;
        const double f = std::log(10000.0) / (double)(half - 1);
        std::vector<float> emb((size_t)dim);
        for (int j = 0; j < half; ++j) {
            const double v = w * std::exp(-f * (double)j);
            emb[(size_t)j] = (float)std::sin(v);
            emb[(size_t)(j + half)] = (float)std::cos(v);
        }
        return emb;
    }

  private:
    LcmConfig cfg_;
    std::vector<double> acp_;
    std::vector<double> timesteps_;
    int step_index_ = 0;
};

} // namespace xllama::diffusion
