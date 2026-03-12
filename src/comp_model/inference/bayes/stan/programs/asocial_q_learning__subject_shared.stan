data {
  int<lower=1> A;
  int<lower=1> T;
  array[T] int<lower=0,upper=A> choice;
  vector[T] reward;
  array[T] vector<lower=0,upper=1>[A] avail_mask;
  array[T] int block_of_trial;

  real alpha_prior_mu;
  real<lower=0> alpha_prior_sigma;
  real beta_prior_mu;
  real<lower=0> beta_prior_sigma;
}
parameters {
  real alpha_z;
  real beta_z;
}
transformed parameters {
  real<lower=0,upper=1> alpha = inv_logit(alpha_z);
  real<lower=0> beta = log1p_exp(beta_z);
}
model {
  vector[A] Q = rep_vector(0.5, A);

  alpha_z ~ normal(alpha_prior_mu, alpha_prior_sigma);
  beta_z ~ normal(beta_prior_mu, beta_prior_sigma);

  for (t in 1:T) {
    if (choice[t] > 0) {
      vector[A] u = beta * Q;
      for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[t] | u);
    }

    if (choice[t] > 0) {
      int a = choice[t];
      Q[a] = Q[a] + alpha * (reward[t] - Q[a]);
    }
  }
}
generated quantities {
  vector[T] log_lik = rep_vector(0.0, T);
  {
    vector[A] Q = rep_vector(0.5, A);

    for (t in 1:T) {
      if (choice[t] > 0) {
        vector[A] u = beta * Q;
        for (a in 1:A) if (avail_mask[t][a] == 0) u[a] = negative_infinity();
        log_lik[t] = categorical_logit_lpmf(choice[t] | u);
      }
      if (choice[t] > 0) {
        int a = choice[t];
        Q[a] = Q[a] + alpha * (reward[t] - Q[a]);
      }
    }
  }
}

