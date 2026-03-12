data {
  real alpha_shared_z;
  real beta_shared_z;
  real alpha_delta_z_social;
  real beta_delta_z_social;
}
generated quantities {
  real alpha_baseline = alpha_shared_z;
  real beta_baseline = beta_shared_z;
  real alpha_social = alpha_shared_z + alpha_delta_z_social;
  real beta_social = beta_shared_z + beta_delta_z_social;
}

