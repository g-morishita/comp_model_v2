data {
  int<lower=1> A;
  int<lower=1> T;
  array[T] int<lower=0,upper=A> choice;
  vector[T] reward;
  array[T] vector<lower=0,upper=1>[A] avail_mask;
  real<lower=0,upper=1> alpha;
  real<lower=0> beta;
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

