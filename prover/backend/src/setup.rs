use lambdaworks_groth16::common::{G1Point, G2Point};
use ndarray::{Array2, Array3, concatenate, Axis,s};
use wasmer_wasix::types::wasi::Errno;
use zkp_rust_tools_math::bipolynomial::BivariatePolynomial;

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve,
                default_types::{FrConfig, FrElement, FrField},
                pairing::BLS12381AtePairing,
                twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
    msm::pippenger::msm,

};

use crate::error::Error;

#[derive(Clone, Debug)]
pub struct Sigma<'a , G1Point, G2Point> {
    pub sigma_a_i:Sigma_A_I<G1Point>, 
    pub sigma_c : Box<Sigma_C<G1Point>>, 
    pub sigma_zk :&'a Sigma_ZK<G1Point>,
    pub sigma_v : &'a Sigma_V<G2Point>,
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


impl Sigma<'_ ,<BLS12381AtePairing as IsPairing>::G1Point,
            <BLS12381AtePairing as IsPairing>::G2Point > 
{   

    // input of this function is d_j(y) * u_j(x) as a bivariate polynimial , 
    pub fn calculate_u1(&self ,poly :&BivariatePolynomial<FrElement>)-> Result<G1Point, Error> {
        // I assume that the dimensions of poly and sigma match, so we don't need flattening
        let coefficients_x_y: Vec<_> = poly.flatten_out()
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();

        let mut result = msm(
            &coefficients_x_y,
            // self.sigma_a_i.x_h_y_i.into_raw_vec_and_offset().0,
            &self.sigma_a_i.x_h_y_i.clone().into_raw_vec_and_offset().0,
        )?;
        result = result.operate_with(&self.sigma_a_i.alpha);
        Ok(result)
    }

    
    // input of this function is d_j(y) * u_j(x) as a bivariate polynimial , 
    pub fn calculate_v2(&self, poly: BivariatePolynomial<FrElement>) -> Result<G2Point,Error> {
        // I assume that the dimensions of poly and sigma match, so we don't need flattening
        let coefficients_x_y: Vec<_> = poly.flatten_out()
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        let mut result = msm(
            &coefficients_x_y,
            // self.sigma_a_i.x_h_y_i.into_raw_vec_and_offset().0,
            &self.sigma_v.x_h_y_i.clone().into_raw_vec_and_offset().0,
        )?;
        result = result.operate_with(&self.sigma_v.beta);
        Ok(result)        
    }


    // pub fn 
}