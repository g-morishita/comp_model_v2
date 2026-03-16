data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> T;
  array[T] int<lower=1,upper=N> subj;
  array[T] int<lower=0,upper=A> choice;
  vector[T] reward;
  array[T] vector<lower=0,upper=1>[A] avail_mask;
  array[T] int block_of_trial;
  int<lower=0,upper=1> reset_on_block;

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[T] int<lower=1,upper=C> cond;

  real alpha_prior_mu;
  real<lower=0> alpha_prior_sigma;
  real beta_prior_mu;
  real<lower=0> beta_prior_sigma;
}
parameters {
  // Population-level: shared
  real mu_alpha_shared_z;
  real<lower=0> sd_alpha_shared_z;
  real mu_beta_shared_z;
  real<lower=0> sd_beta_shared_z;

  // Population-level: deltas
  vector[C - 1] mu_alpha_delta_z;
  vector<lower=0>[C - 1] sd_alpha_delta_z;
  vector[C - 1] mu_beta_delta_z;
  vector<lower=0>[C - 1] sd_beta_delta_z;

  // Per-subject raw (non-centered)
  vector[N] raw_alpha_shared_z;
  vector[N] raw_beta_shared_z;
  array[C - 1] vector[N] raw_alpha_delta_z;
  array[C - 1] vector[N] raw_beta_delta_z;
}
transformed parameters {
  // Per-subject unconstrained shared
  vector[N] alpha_shared_z = mu_alpha_shared_z + sd_alpha_shared_z * raw_alpha_shared_z;
  vector[N] beta_shared_z = mu_beta_shared_z + sd_beta_shared_z * raw_beta_shared_z;

  // Per-subject unconstrained deltas
  array[C - 1] vector[N] alpha_delta_z;
  array[C - 1] vector[N] beta_delta_z;
  for (d in 1:(C - 1)) {
    alpha_delta_z[d] = mu_alpha_delta_z[d] + sd_alpha_delta_z[d] * raw_alpha_delta_z[d];
    beta_delta_z[d] = mu_beta_delta_z[d] + sd_beta_delta_z[d] * raw_beta_delta_z[d];
  }

  // Per-subject, per-condition constrained parameters
  array[N] vector<lower=0,upper=1>[C] alpha;
  array[N] vector<lower=0>[C] beta;
  for (n in 1:N) {
    int d = 0;
    for (c in 1:C) {
      real az = alpha_shared_z[n];
      real bz = beta_shared_z[n];
      if (c != baseline_cond) {
        d += 1;
        az += alpha_delta_z[d][n];
        bz += beta_delta_z[d][n];
      }
      alpha[n][c] = inv_logit(az);
      beta[n][c] = log1p_exp(bz);
    }
  }
}
model {
  array[N] vector[A] Q;

  // Priors: shared
  mu_alpha_shared_z ~ normal(alpha_prior_mu, alpha_prior_sigma);
  sd_alpha_shared_z ~ normal(0, 1);
  raw_alpha_shared_z ~ normal(0, 1);

  mu_beta_shared_z ~ normal(beta_prior_mu, beta_prior_sigma);
  sd_beta_shared_z ~ normal(0, 1);
  raw_beta_shared_z ~ normal(0, 1);

  // Priors: deltas
  mu_alpha_delta_z ~ normal(0, 1);
  sd_alpha_delta_z ~ normal(0, 1);
  mu_beta_delta_z ~ normal(0, 1);
  sd_beta_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1)) {
    raw_alpha_delta_z[d] ~ normal(0, 1);
    raw_beta_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) Q[n] = rep_vector(0.5, A);

  for (t in 1:T) {
    int n = subj[t];
    if (reset_on_block == 1 && t > 1 && subj[t] == subj[t - 1] &&
        block_of_trial[t] != block_of_trial[t - 1]) {
      Q[n] = rep_vector(0.5, A);
    }
    if (choice[t] > 0) {
      int cc = cond[t];
      vector[A] u = beta[n][cc] * Q[n];
      for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[t] | u);
    }
    if (choice[t] > 0) {
      int a = choice[t];
      int cc = cond[t];
      Q[n][a] = Q[n][a] + alpha[n][cc] * (reward[t] - Q[n][a]);
    }
  }
}
generated quantities {
  vector[T] log_lik = rep_vector(0.0, T);
  {
    array[N] vector[A] Q;
    for (n in 1:N) Q[n] = rep_vector(0.5, A);

    for (t in 1:T) {
      int n = subj[t];
      if (reset_on_block == 1 && t > 1 && subj[t] == subj[t - 1] &&
          block_of_trial[t] != block_of_trial[t - 1]) {
        Q[n] = rep_vector(0.5, A);
      }
      if (choice[t] > 0) {
        int cc = cond[t];
        vector[A] u = beta[n][cc] * Q[n];
        for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
        log_lik[t] = categorical_logit_lpmf(choice[t] | u);
      }
      if (choice[t] > 0) {
        int a = choice[t];
        int cc = cond[t];
        Q[n][a] = Q[n][a] + alpha[n][cc] * (reward[t] - Q[n][a]);
      }
    }
  }

  // Population-level constrained parameters (baseline condition)
  real alpha_shared_pop = inv_logit(mu_alpha_shared_z);
  real beta_shared_pop = log1p_exp(mu_beta_shared_z);
}
