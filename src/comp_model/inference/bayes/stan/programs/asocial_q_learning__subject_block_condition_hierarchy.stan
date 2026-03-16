data {
  int<lower=1> A;
  int<lower=1> T;
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
  real alpha_shared_z;
  real beta_shared_z;
  vector[C - 1] alpha_delta_z;
  vector[C - 1] beta_delta_z;
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha;
  vector<lower=0>[C] beta;
  {
    int d = 0;
    for (c in 1:C) {
      real az = alpha_shared_z;
      real bz = beta_shared_z;
      if (c != baseline_cond) {
        d += 1;
        az += alpha_delta_z[d];
        bz += beta_delta_z[d];
      }
      alpha[c] = inv_logit(az);
      beta[c] = log1p_exp(bz);
    }
  }
}
model {
  vector[A] Q = rep_vector(0.5, A);

  alpha_shared_z ~ normal(alpha_prior_mu, alpha_prior_sigma);
  beta_shared_z ~ normal(beta_prior_mu, beta_prior_sigma);
  alpha_delta_z ~ normal(0, 1);
  beta_delta_z ~ normal(0, 1);

  for (t in 1:T) {
    if (reset_on_block == 1 && t > 1 && block_of_trial[t] != block_of_trial[t - 1]) {
      Q = rep_vector(0.5, A);
    }
    if (choice[t] > 0) {
      int cc = cond[t];
      vector[A] u = beta[cc] * Q;
      for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[t] | u);
    }
    if (choice[t] > 0) {
      int a = choice[t];
      int cc = cond[t];
      Q[a] = Q[a] + alpha[cc] * (reward[t] - Q[a]);
    }
  }
}
generated quantities {
  vector[T] log_lik = rep_vector(0.0, T);
  {
    vector[A] Q = rep_vector(0.5, A);

    for (t in 1:T) {
      if (reset_on_block == 1 && t > 1 && block_of_trial[t] != block_of_trial[t - 1]) {
        Q = rep_vector(0.5, A);
      }
      if (choice[t] > 0) {
        int cc = cond[t];
        vector[A] u = beta[cc] * Q;
        for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
        log_lik[t] = categorical_logit_lpmf(choice[t] | u);
      }
      if (choice[t] > 0) {
        int a = choice[t];
        int cc = cond[t];
        Q[a] = Q[a] + alpha[cc] * (reward[t] - Q[a]);
      }
    }
  }
}
