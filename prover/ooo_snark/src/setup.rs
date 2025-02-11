use lambdaworks_groth16::common::G1Point;
use ndarray::{Array2, Array3, concatenate, Axis,s};



#[derive(Clone, Debug)]
pub struct Sigma<G1Point, G2Point> {
    pub sigma_a_i :Sigma_A_I<G1Point>, 
    pub sigma_c : Sigma_C<G1Point>, 
    pub sigma_zk :Sigma_ZK<G1Point>,
    pub sigma_v : Sigma_V<G2Point>,
}

#[derive(Clone, Debug)]
pub struct Sigma_A_I<G1Point> {
    // first line paper page 21 
    pub alpha: G1Point,  
    pub x_h_y_i: Array2<G1Point>, // h ∈ ⟦0,n-1⟧ , i ∈ ⟦0, s ₍max₎ -1⟧
    // second line paper page 21
    pub gamma_inv_l0_y_oj_x :Vec<G1Point>, // j ∈ ⟦0,l ₍in₎-1⟧
    pub gamma_inv_lsmax_minus_1_oj_x :Vec<G1Point>, //  j ∈ ⟦0, l-1⟧
    pub eta1_inv_li_y_oj_x :Array2<G1Point>, // i ∈ ⟦0, s ₍max₎ -1⟧ , j ∈ ⟦l, l₍D₎ - 1⟧
    pub delta_inv_li_y_oj_x :Array2<G1Point>,  // i ∈ ⟦0, s ₍max₎ -1⟧ , j ∈ ⟦l₍D₎ , m₍D₎ - 1 ⟧
    // third line paper page 21 
    pub eta0_inv_li_y_oj_x_paranth_k_j_minus_l_pow2_z_minus_1_paranth: Array2<G1Point>, //i ∈ ⟦0, s ₍max₎ -1⟧ , j ∈ ⟦l₍D₎ , m₍D₎ - 1 ⟧
    // fourth line paper page 21 
    pub delta_inv_x_h_y_i_tx_x :Array2<G1Point>, // h ∈ ⟦0, 2n-2⟧ , i ∈ ⟦0 , s₍max₎ - 2 ⟧
    pub delta_inv_x_h_y_i_ty_y :Array2<G1Point>, // h ∈ ⟦0, 2n-2⟧ , i ∈ ⟦0 , s₍max₎ - 2 ⟧
    pub eta0_inv_li_y_mj_x_z_tz_z: Array2<G1Point> , // i ∈ ⟦0 , s₍max₎ - 1 ⟧ , j ∈ ⟦l , l₍D₎ - 1 ⟧
}   

#[derive(Clone, Debug)]
pub struct Sigma_C<G1Point> {
    // first line paper page 21
    pub mu_inv_li_y_kj_z: Array2<G1Point>,// i ∈ ⟦0 , s₍max₎ - 1 ⟧ , j ∈ ⟦0 , l₍D₎ - l - 1 ⟧
    // second line paper page 21 
    pub nu_inv_y_i_z_j_ty_y :Array2<G1Point>,  // i ∈ ⟦0 , s₍max₎ - 2 ⟧ , j ∈ ⟦0 , 2⋅l₍D₎ - 2⋅l - 2 ⟧
    pub nu_inv_y_i_z_j_tz_z :Array2<G1Point>,  // i ∈ ⟦0 , 2⋅s₍max₎ - 2 ⟧ , j ∈ ⟦0 , 2⋅l₍D₎ - 2⋅l - 3 ⟧
    // third line paper page 21 
    pub psi0_inv_kappa_h_y_i_z_j :Array3<G1Point>, // h ∈ ⟦0 , 1⟧ ,  i ∈ ⟦0 , 2⋅s₍max₎ - 3 ⟧,j ∈ ⟦0 , 3⋅l₍D₎ - 3⋅l - 3 ⟧
    pub psi1_inv_z_j :Vec<G1Point>,// j ∈ ⟦0 , 3⋅l₍D₎ - 3⋅l - 4 ⟧
    pub psi2_inv_kappa_2_y_i_z_j :Array2<G1Point> , //i ∈ ⟦0 , 2⋅s₍max₎ - 2 ⟧,j ∈ ⟦0 , l₍D₎ - l - 1 ⟧
    pub psi3_inv_kappa_h_z_j :Array2<G1Point>, // h ∈ ⟦1 , 2⟧, j ∈ ⟦0 , l₍D₎ - l - 2 ⟧
}

#[derive(Clone, Debug)]
pub struct Sigma_ZK<G1Point>{
    // first line paper page 21
    pub beta :G1Point, 
    pub delta :G1Point, 
    pub eta1 :G1Point, 
    pub mu_inv_y_j_ty_y :Vec<G1Point>, //i ∈ ⟦0 , 1⟧ 
    pub eta0_inv_ty_y_sum_j_equal_l_till_ld_minus_1_mj_x_z_tz_z :G1Point, 
    // second line paper page 21 
    pub eta1_inv_ty_y_sum_j_equal_l_till_ld_minus_1_oj_x :G1Point, 
    pub eta0_inv_ty_y_sum_j_equal_l_till_ld_minus_1_oj_x_paranth_k_j_minus_l_pow2_z_minus_1_paranth : G1Point, 
    // third line paper 21 
    pub nu_inv_y_i_z_j_ty_y :Array2<G1Point>, // i ∈ ⟦s₍max₎ - 1 , s₍max₎ + 1 ⟧ , j ∈ ⟦0 , 2⋅l₍D₎ - 2⋅l - 2 ⟧
    pub psi0_inv_kappa_h_y_i_z_j :Array3<G1Point>, // h ∈ ⟦0 , 1⟧ ,  i ∈ ⟦2⋅s₍max₎ - 2,  2⋅s₍max₎⟧,j ∈ ⟦0 , 3⋅l₍D₎ - 3⋅l - 3 ⟧
    pub psi2_inv_kappa_2_y_i_z_j :Array2<G1Point> , //i ∈ ⟦s₍max₎ - 1, s₍max₎ ⟧,j ∈ ⟦0 , l₍D₎ - l - 1 ⟧

}

#[derive(Clone, Debug)]
pub struct Sigma_V<G2Point>{
    // first line paper page 21
    pub beta :G2Point, 
    pub gamma :G2Point, 
    pub delta :G2Point, 
    pub eta1 :G2Point,  
    pub mu_eta0 :G2Point, 
    pub mu_eta1 :G2Point, 
    pub x_h_y_i: Array2<G2Point>, // h ∈ ⟦0,n-1⟧ , i ∈ ⟦0, s ₍max₎ -1⟧
    pub mu_2_sum_j_equal_l_till_ld_minus_1_oj_x_k_j_minus_l : G1Point, 
    // third line paper 21 
    pub mu_3_nu :G2Point, 
    pub mu_4_kappa_h :Vec<G2Point>, // h ∈ ⟦0,2⟧ 
    // third line last element
    pub mu_3_psi_0_y_i_z_j :Vec<G2Point>,//i ∈ ⟦0, 1⟧, j ∈ ⟦0, 1⟧
    pub mu_3_psi_1_y_i_z_j :Vec<G2Point>,//i ∈ ⟦0, 1⟧, j ∈ ⟦0, 1⟧
    pub mu_3_psi_2_y_i_z_j :Vec<G2Point>,//i ∈ ⟦0, 1⟧, j ∈ ⟦0, 1⟧
    pub mu_3_psi_3_y_i_z_j :Vec<G2Point>,//i ∈ ⟦0, 1⟧, j ∈ ⟦0, 1⟧

}