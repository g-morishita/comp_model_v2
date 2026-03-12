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

  real alpha_prior_mu;
  real<lower=0> alpha_prior_sigma;
  real beta_prior_mu;
  real<lower=0> beta_prior_sigma;
}
parameters {
  real mu_alpha_z;
  real<lower=0> sd_alpha_z;
  vector[N] raw_alpha_z;

  real mu_beta_z;
  real<lower=0> sd_beta_z;
  vector[N] raw_beta_z;
}
transformed parameters {
  vector[N] alpha_z = mu_alpha_z + sd_alpha_z * raw_alpha_z;
  vector[N] beta_z = mu_beta_z + sd_beta_z * raw_beta_z;
  vector<lower=0,upper=1>[N] alpha = inv_logit(alpha_z);
  vector<lower=0>[N] beta = log1p_exp(beta_z);
}
model {
  array[N] vector[A] Q;

  mu_alpha_z ~ normal(alpha_prior_mu, alpha_prior_sigma);
  sd_alpha_z ~ normal(0, 1);
  raw_alpha_z ~ normal(0, 1);

  mu_beta_z ~ normal(beta_prior_mu, beta_prior_sigma);
  sd_beta_z ~ normal(0, 1);
  raw_beta_z ~ normal(0, 1);

  for (n in 1:N) Q[n] = rep_vector(0.5, A);

  for (t in 1:T) {
    int n = subj[t];
    if (reset_on_block == 1 && t > 1 && subj[t] == subj[t - 1] &&
        block_of_trial[t] != block_of_trial[t - 1]) {
      Q[n] = rep_vector(0.5, A);
    }
    if (choice[t] > 0) {
      vector[A] u = beta[n] * Q[n];
      for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[t] | u);
    }
    if (choice[t] > 0) {
      int a = choice[t];
      Q[n][a] = Q[n][a] + alpha[n] * (reward[t] - Q[n][a]);
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
        vector[A] u = beta[n] * Q[n];
        for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
        log_lik[t] = categorical_logit_lpmf(choice[t] | u);
      }
      if (choice[t] > 0) {
        int a = choice[t];
        Q[n][a] = Q[n][a] + alpha[n] * (reward[t] - Q[n][a]);
      }
    }
  }
  real alpha_pop = inv_logit(mu_alpha_z);
  real beta_pop = log1p_exp(mu_beta_z);
}
