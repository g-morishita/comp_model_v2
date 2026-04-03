/*
 * Model: Asocial RL — Asymmetric Learning Rates
 * Hierarchy: Study-subject-block-condition (multiple subjects, per-subject per-condition parameters)
 * Parameters: alpha_pos[N][C], alpha_neg[N][C], beta[N][C] — per-subject per-condition, via group-level shared + delta hierarchies
 *
 * Two-level hierarchical asymmetric Q-learning. Each subject has condition-specific alpha_pos,
 * alpha_neg, and beta built from a subject-level baseline plus a condition delta, both drawn from
 * their own group distributions (non-centred parameterisation).
 */
functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> N;                                    // number of subjects
  int<lower=1> A;                                    // number of available actions
  int<lower=1> E;                                    // total number of steps (DECISION + UPDATE events)
  int<lower=0> D;                                    // number of DECISION steps (used to size log_lik)
  array[E] int<lower=1,upper=N> step_subject;        // subject index for each step
  array[E] int<lower=0,upper=A> step_choice;         // chosen action at step e; 0 if no choice (UPDATE-only step)
  array[E] int<lower=0,upper=A> step_update_action;  // action whose Q-value is updated at step e; 0 if no update
  vector[E] step_reward;                             // reward received at the update step
  array[E] vector<lower=0,upper=1>[A] step_avail_mask; // binary mask of available actions (1 = available)
  array[E] int step_block;                           // block index for each step (used to detect block boundaries)
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q-values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  int<lower=2> C;                        // number of experimental conditions
  int<lower=1,upper=C> baseline_cond;    // index of the baseline condition (receives no delta)
  array[E] int<lower=1,upper=C> step_condition; // condition index for each step

  int alpha_pos_prior_family;   // prior family code for the group-level shared alpha_pos mean
  real alpha_pos_prior_p1;      // first hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p2;      // second hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p3;      // third hyperparameter of the alpha_pos prior
  int alpha_pos_delta_prior_family;   // prior family code for the group-level alpha_pos delta means
  real alpha_pos_delta_prior_p1;      // first hyperparameter of the alpha_pos delta prior
  real alpha_pos_delta_prior_p2;      // second hyperparameter of the alpha_pos delta prior
  real alpha_pos_delta_prior_p3;      // third hyperparameter of the alpha_pos delta prior
  int alpha_neg_prior_family;   // prior family code for the group-level shared alpha_neg mean
  real alpha_neg_prior_p1;      // first hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p2;      // second hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p3;      // third hyperparameter of the alpha_neg prior
  int alpha_neg_delta_prior_family;   // prior family code for the group-level alpha_neg delta means
  real alpha_neg_delta_prior_p1;      // first hyperparameter of the alpha_neg delta prior
  real alpha_neg_delta_prior_p2;      // second hyperparameter of the alpha_neg delta prior
  real alpha_neg_delta_prior_p3;      // third hyperparameter of the alpha_neg delta prior
  int beta_prior_family;        // prior family code for the group-level shared beta mean
  real beta_prior_p1;           // first hyperparameter of the beta prior
  real beta_prior_p2;           // second hyperparameter of the beta prior
  real beta_prior_p3;           // third hyperparameter of the beta prior
  int beta_delta_prior_family;        // prior family code for the group-level beta delta means
  real beta_delta_prior_p1;           // first hyperparameter of the beta delta prior
  real beta_delta_prior_p2;           // second hyperparameter of the beta delta prior
  real beta_delta_prior_p3;           // third hyperparameter of the beta delta prior
  int sd_alpha_pos_prior_family;   // prior family code for the group-level shared alpha_pos SD
  real sd_alpha_pos_prior_p1;      // first hyperparameter of the shared alpha_pos SD prior
  real sd_alpha_pos_prior_p2;      // second hyperparameter of the shared alpha_pos SD prior
  real sd_alpha_pos_prior_p3;      // third hyperparameter of the shared alpha_pos SD prior
  int sd_alpha_neg_prior_family;   // prior family code for the group-level shared alpha_neg SD
  real sd_alpha_neg_prior_p1;      // first hyperparameter of the shared alpha_neg SD prior
  real sd_alpha_neg_prior_p2;      // second hyperparameter of the shared alpha_neg SD prior
  real sd_alpha_neg_prior_p3;      // third hyperparameter of the shared alpha_neg SD prior
  int sd_beta_prior_family;        // prior family code for the group-level shared beta SD
  real sd_beta_prior_p1;           // first hyperparameter of the shared beta SD prior
  real sd_beta_prior_p2;           // second hyperparameter of the shared beta SD prior
  real sd_beta_prior_p3;           // third hyperparameter of the shared beta SD prior
  int sd_alpha_pos_delta_prior_family;   // prior family code for the group-level alpha_pos delta SD
  real sd_alpha_pos_delta_prior_p1;      // first hyperparameter of the alpha_pos delta SD prior
  real sd_alpha_pos_delta_prior_p2;      // second hyperparameter of the alpha_pos delta SD prior
  real sd_alpha_pos_delta_prior_p3;      // third hyperparameter of the alpha_pos delta SD prior
  int sd_alpha_neg_delta_prior_family;   // prior family code for the group-level alpha_neg delta SD
  real sd_alpha_neg_delta_prior_p1;      // first hyperparameter of the alpha_neg delta SD prior
  real sd_alpha_neg_delta_prior_p2;      // second hyperparameter of the alpha_neg delta SD prior
  real sd_alpha_neg_delta_prior_p3;      // third hyperparameter of the alpha_neg delta SD prior
  int sd_beta_delta_prior_family;        // prior family code for the group-level beta delta SD
  real sd_beta_delta_prior_p1;           // first hyperparameter of the beta delta SD prior
  real sd_beta_delta_prior_p2;           // second hyperparameter of the beta delta SD prior
  real sd_beta_delta_prior_p3;           // third hyperparameter of the beta delta SD prior
}
parameters {
  // Population-level: shared
  real mu_alpha_pos_shared_z;              // group mean of baseline alpha_pos (unconstrained)
  real<lower=0> sd_alpha_pos_shared_z;     // group SD of baseline alpha_pos (unconstrained)
  real mu_alpha_neg_shared_z;              // group mean of baseline alpha_neg (unconstrained)
  real<lower=0> sd_alpha_neg_shared_z;     // group SD of baseline alpha_neg (unconstrained)
  real mu_beta_shared_z;                   // group mean of baseline beta (unconstrained)
  real<lower=0> sd_beta_shared_z;          // group SD of baseline beta (unconstrained)

  // Population-level: deltas
  vector[C - 1] mu_alpha_pos_delta_z;          // group means of per-condition alpha_pos deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_pos_delta_z; // group SDs of per-condition alpha_pos deltas
  vector[C - 1] mu_alpha_neg_delta_z;          // group means of per-condition alpha_neg deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_neg_delta_z; // group SDs of per-condition alpha_neg deltas
  vector[C - 1] mu_beta_delta_z;               // group means of per-condition beta deltas (unconstrained)
  vector<lower=0>[C - 1] sd_beta_delta_z;      // group SDs of per-condition beta deltas

  // Per-subject raw (non-centered)
  vector[N] raw_alpha_pos_shared_z;            // per-subject deviates for baseline alpha_pos
  vector[N] raw_alpha_neg_shared_z;            // per-subject deviates for baseline alpha_neg
  vector[N] raw_beta_shared_z;                 // per-subject deviates for baseline beta
  array[C - 1] vector[N] raw_alpha_pos_delta_z; // per-subject deviates for alpha_pos condition deltas
  array[C - 1] vector[N] raw_alpha_neg_delta_z; // per-subject deviates for alpha_neg condition deltas
  array[C - 1] vector[N] raw_beta_delta_z;      // per-subject deviates for beta condition deltas
}
transformed parameters {
  vector[N] alpha_pos_shared_z = mu_alpha_pos_shared_z + sd_alpha_pos_shared_z * raw_alpha_pos_shared_z; // subject baseline alpha_pos (unconstrained)
  vector[N] alpha_neg_shared_z = mu_alpha_neg_shared_z + sd_alpha_neg_shared_z * raw_alpha_neg_shared_z; // subject baseline alpha_neg (unconstrained)
  vector[N] beta_shared_z = mu_beta_shared_z + sd_beta_shared_z * raw_beta_shared_z;                     // subject baseline beta (unconstrained)

  array[C - 1] vector[N] alpha_pos_delta_z; // per-subject condition alpha_pos deltas (unconstrained)
  array[C - 1] vector[N] alpha_neg_delta_z; // per-subject condition alpha_neg deltas (unconstrained)
  array[C - 1] vector[N] beta_delta_z;      // per-subject condition beta deltas (unconstrained)
  for (d in 1:(C - 1)) {
    alpha_pos_delta_z[d] = mu_alpha_pos_delta_z[d] + sd_alpha_pos_delta_z[d] * raw_alpha_pos_delta_z[d];
    alpha_neg_delta_z[d] = mu_alpha_neg_delta_z[d] + sd_alpha_neg_delta_z[d] * raw_alpha_neg_delta_z[d];
    beta_delta_z[d] = mu_beta_delta_z[d] + sd_beta_delta_z[d] * raw_beta_delta_z[d];
  }

  array[N] vector<lower=0,upper=1>[C] alpha_pos; // per-subject per-condition positive learning rate in (0,1)
  array[N] vector<lower=0,upper=1>[C] alpha_neg; // per-subject per-condition negative learning rate in (0,1)
  array[N] vector<lower=0>[C] beta;              // per-subject per-condition inverse temperature > 0
  for (n in 1:N) {
    int d = 0; // delta index counter
    for (c in 1:C) {
      real apz = alpha_pos_shared_z[n]; // start from subject's baseline
      real anz = alpha_neg_shared_z[n];
      real bz = beta_shared_z[n];
      // Non-baseline conditions add the subject's condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        apz += alpha_pos_delta_z[d][n];
        anz += alpha_neg_delta_z[d][n];
        bz += beta_delta_z[d][n];
      }
      alpha_pos[n][c] = inv_logit(apz); // map to (0,1)
      alpha_neg[n][c] = inv_logit(anz);
      beta[n][c] = log1p_exp(bz);       // map to (0,inf)
    }
  }
}
model {
  array[N] vector[A] Q; // per-subject action-value vectors

  target += prior_lpdf(mu_alpha_pos_shared_z | alpha_pos_prior_family, alpha_pos_prior_p1, alpha_pos_prior_p2, alpha_pos_prior_p3);
  target += prior_lpdf(sd_alpha_pos_shared_z | sd_alpha_pos_prior_family, sd_alpha_pos_prior_p1, sd_alpha_pos_prior_p2, sd_alpha_pos_prior_p3);
  raw_alpha_pos_shared_z ~ normal(0, 1);  // non-centred parameterisation

  target += prior_lpdf(mu_alpha_neg_shared_z | alpha_neg_prior_family, alpha_neg_prior_p1, alpha_neg_prior_p2, alpha_neg_prior_p3);
  target += prior_lpdf(sd_alpha_neg_shared_z | sd_alpha_neg_prior_family, sd_alpha_neg_prior_p1, sd_alpha_neg_prior_p2, sd_alpha_neg_prior_p3);
  raw_alpha_neg_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(sd_beta_shared_z | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);
  raw_beta_shared_z ~ normal(0, 1);

  for (d in 1:(C - 1))
    target += prior_lpdf(mu_alpha_pos_delta_z[d] | alpha_pos_delta_prior_family, alpha_pos_delta_prior_p1, alpha_pos_delta_prior_p2, alpha_pos_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_pos_delta_z[d] | sd_alpha_pos_delta_prior_family, sd_alpha_pos_delta_prior_p1, sd_alpha_pos_delta_prior_p2, sd_alpha_pos_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_alpha_neg_delta_z[d] | alpha_neg_delta_prior_family, alpha_neg_delta_prior_p1, alpha_neg_delta_prior_p2, alpha_neg_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_neg_delta_z[d] | sd_alpha_neg_delta_prior_family, sd_alpha_neg_delta_prior_p1, sd_alpha_neg_delta_prior_p2, sd_alpha_neg_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_beta_delta_z[d] | beta_delta_prior_family, beta_delta_prior_p1, beta_delta_prior_p2, beta_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_beta_delta_z[d] | sd_beta_delta_prior_family, sd_beta_delta_prior_p1, sd_beta_delta_prior_p2, sd_beta_delta_prior_p3);
  for (d in 1:(C - 1)) {
    raw_alpha_pos_delta_z[d] ~ normal(0, 1); // non-centred deviates for per-subject alpha_pos deltas
    raw_alpha_neg_delta_z[d] ~ normal(0, 1);
    raw_beta_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) Q[n] = rep_vector(q_init, A); // initialise Q-values for all subjects

  for (e in 1:E) {
    int n = step_subject[e]; // subject for this step
    // Reset this subject's Q-values when their block changes
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability using subject n's condition-specific beta
    if (step_choice[e] > 0) {
      int cc = step_condition[e];                                                // condition at this step
      vector[A] u = beta[n][cc] * Q[n];                                         // scale by subject's condition beta
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: asymmetric update with subject n's condition-specific alphas
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      real delta = step_reward[e] - Q[n][a];                                                           // prediction error
      Q[n][a] = Q[n][a] + (delta >= 0 ? alpha_pos[n][cc] : alpha_neg[n][cc]) * delta; // branch on sign of PE
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for WAIC model comparison
  vector<lower=0,upper=1>[C] alpha_pos_pop; // group-mean alpha_pos for every condition (constrained)
  vector<lower=0,upper=1>[C] alpha_neg_pop; // group-mean alpha_neg for every condition (constrained)
  vector<lower=0>[C] beta_pop;              // group-mean beta for every condition (constrained)
  real alpha_pos_shared_pop;                // group-mean baseline alpha_pos (constrained)
  real alpha_neg_shared_pop;                // group-mean baseline alpha_neg (constrained)
  real beta_shared_pop;                     // group-mean baseline beta (constrained)
  {
    array[N] vector[A] Q;
    for (n in 1:N) Q[n] = rep_vector(q_init, A);
    int d = 0; // decision counter (indexes into log_lik)

    for (e in 1:E) {
      int n = step_subject[e];
      // Reset this subject's Q-values when their block changes
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[n][cc] * Q[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: asymmetric update with subject n's condition-specific alphas
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        real delta = step_reward[e] - Q[n][a];
        Q[n][a] = Q[n][a] + (delta >= 0 ? alpha_pos[n][cc] : alpha_neg[n][cc]) * delta;
      }
    }
  }

  {
    int d_idx = 0;
    for (c in 1:C) {
      real apz = mu_alpha_pos_shared_z;
      real anz = mu_alpha_neg_shared_z;
      real bz = mu_beta_shared_z;
      if (c != baseline_cond) {
        d_idx += 1;
        apz += mu_alpha_pos_delta_z[d_idx];
        anz += mu_alpha_neg_delta_z[d_idx];
        bz += mu_beta_delta_z[d_idx];
      }
      alpha_pos_pop[c] = inv_logit(apz);
      alpha_neg_pop[c] = inv_logit(anz);
      beta_pop[c] = log1p_exp(bz);
    }
  }

  alpha_pos_shared_pop = alpha_pos_pop[baseline_cond];
  alpha_neg_shared_pop = alpha_neg_pop[baseline_cond];
  beta_shared_pop = beta_pop[baseline_cond];
}
