data {
  int<lower=1> A;
  int<lower=1> T;
  array[T] int<lower=0,upper=A> choice;
  vector[T] reward;
  array[T] vector<lower=0,upper=1>[A] avail_mask;
  array[T] int block_of_trial;
  int<lower=0,upper=1> reset_on_block;
  real<lower=0,upper=1> alpha;
  real<lower=0> beta;
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
