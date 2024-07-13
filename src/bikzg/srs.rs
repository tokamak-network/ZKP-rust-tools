use std::ops::Mul;

use core::{marker::PhantomData, mem};

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{curves::bls12_381::{curve::BLS12381Curve, default_types::FrElement}, point::ShortWeierstrassProjectivePoint},
        traits::IsEllipticCurve,
    },
    traits::{AsBytes, Deserializable},
    errors::DeserializationError,
    field::{element::FieldElement, traits::IsPrimeField},
    elliptic_curve::traits::IsPairing,
    msm::pippenger::msm,
    unsigned_integer::element::UnsignedInteger,
};

use crate::bikzg::traits::IsCommitmentScheme;

use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;

use rayon::prelude::*;

use crate::{bipolynomial::BivariatePolynomial, G1Point};

#[derive(PartialEq, Clone, Debug)]
pub struct StructuredReferenceString<G1Point, G2Point> {
    pub dimention_x: usize, 
    pub dimention_y: usize,
    pub powers_main_group: Vec<G1Point>,
    pub powers_secondary_group: [G2Point; 3],// 1 , tau, theta 
}


impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    pub fn new(dim_x: usize, dim_y: usize,powers_main_group: &[G1Point], powers_secondary_group: &[G2Point; 3]) -> Self {
        Self {
            dimention_x: dim_x, 
            dimention_y: dim_y,
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

impl<G1Point, G2Point> AsBytes for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + AsBytes,
    G2Point: IsGroup + AsBytes,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut serialized_data: Vec<u8> = Vec::new();
        // First 4 bytes encodes protocol version
        let protocol_version: [u8; 4] = [0; 4];

        serialized_data.extend(&protocol_version);

        add_usize(& mut serialized_data, self.dimention_x);
        add_usize(& mut serialized_data, self.dimention_y);


        // Second 8 bytes store the amount of G1 elements to be stored, this is more than can be indexed with a 64-bit architecture, and some millions of terabytes of data if the points were compressed
        let mut main_group_len_bytes: Vec<u8> = self.powers_main_group.len().to_le_bytes().to_vec();

        // For architectures with less than 64 bits for pointers
        // We add extra zeros at the end
        while main_group_len_bytes.len() < 8 {
            main_group_len_bytes.push(0)
        }

        serialized_data.extend(&main_group_len_bytes);

        // G1 elements
        for point in &self.powers_main_group {
            serialized_data.extend(point.as_bytes());
        }

        // G2 elements
        for point in &self.powers_secondary_group {
            serialized_data.extend(point.as_bytes());
        }

        serialized_data
    }
}

fn add_usize(serialized_data: & mut Vec<u8>, data: usize) {
    let mut data_bytes: Vec<u8> = data.to_le_bytes().to_vec();
    // For data with less than 64 bits for pointers
    // We add extra zeros at the end`
    while data_bytes.len() < 8 { 
        data_bytes.push(0);
    }
    serialized_data.extend(&data_bytes);
}



impl<G1Point, G2Point> Deserializable for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + Deserializable,
    G2Point: IsGroup + Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        const X_DIMENTION_LEN_START: usize = 4; 
        const X_DIMENTION_LEN_END: usize = 12; 

        const Y_DIMENTION_LEN_START: usize = 12; 
        const Y_DIMENTION_LEN_END: usize = 20; 
   
        const MAIN_GROUP_LEN_OFFSET: usize = 20;
        const MAIN_GROUP_OFFSET: usize = 28;

        let x_dim_len_u64 = u64::from_le_bytes(
            // This unwrap can't fail since we are fixing the size of the slice
            bytes[X_DIMENTION_LEN_START..X_DIMENTION_LEN_END]
                .try_into()
                .unwrap(),
        );

        let x_dim_len = usize::try_from(x_dim_len_u64)
        .map_err(|_| DeserializationError::PointerSizeError)?;

        let y_dim_len_u64 = u64::from_le_bytes(
            // This unwrap can't fail since we are fixing the size of the slice
            bytes[Y_DIMENTION_LEN_START..Y_DIMENTION_LEN_END]
                .try_into()
                .unwrap(),
        );

        let y_dim_len = usize::try_from(y_dim_len_u64)
        .map_err(|_| DeserializationError::PointerSizeError)?;

        let main_group_len_u64 = u64::from_le_bytes(
            // This unwrap can't fail since we are fixing the size of the slice
            bytes[MAIN_GROUP_LEN_OFFSET..MAIN_GROUP_OFFSET]
                .try_into()
                .unwrap(),
        );

        let main_group_len = usize::try_from(main_group_len_u64)
            .map_err(|_| DeserializationError::PointerSizeError)?;

        let mut main_group: Vec<G1Point> = Vec::new();
        let mut secondary_group: Vec<G2Point> = Vec::new();

        let size_g1_point = mem::size_of::<G1Point>();
        let size_g2_point = mem::size_of::<G2Point>();

        for i in 0..main_group_len {
            // The second unwrap shouldn't fail since the amount of bytes is fixed
            let point = G1Point::deserialize(
                bytes[i * size_g1_point + MAIN_GROUP_OFFSET
                    ..i * size_g1_point + size_g1_point + MAIN_GROUP_OFFSET]
                    .try_into()
                    .unwrap(),
            )?;
            main_group.push(point);
        }

        let g2s_offset = size_g1_point * main_group_len + 28;
        for i in 0..3 {
            // The second unwrap shouldn't fail since the amount of bytes is fixed
            let point = G2Point::deserialize(
                bytes[i * size_g2_point + g2s_offset
                    ..i * size_g2_point + g2s_offset + size_g2_point]
                    .try_into()
                    .unwrap(),
            )?;
            secondary_group.push(point);
        }

        let secondary_group_slice = [secondary_group[0].clone(), secondary_group[1].clone(), secondary_group[2].clone()];

        let srs = StructuredReferenceString::new(x_dim_len,y_dim_len,&main_group, &secondary_group_slice);
        Ok(srs)
    }
}


#[derive(Clone)]
pub struct BivariateKateZaveruchaGoldberg<F: IsPrimeField, P: IsPairing> {
    srs: StructuredReferenceString<P::G1Point, P::G2Point>,
    phantom: PhantomData<F>,
}

impl<F: IsPrimeField, P: IsPairing> BivariateKateZaveruchaGoldberg<F, P> {
    pub fn new(srs: StructuredReferenceString<P::G1Point, P::G2Point>) -> Self {
        Self {
            srs,
            phantom: PhantomData,
        }
    }
}
impl<const N: usize, F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>, P: IsPairing>
    IsCommitmentScheme<F> for BivariateKateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;

    //not compeleted , should return 2 commitment, one for q_xy another for q_y
    fn commit(&self, bp: &BivariatePolynomial<FieldElement<F>>, p: &UnivariatePolynomial<FieldElement<F>>) -> (Self::Commitment,Self::Commitment) {
        
        let coefficients_y: Vec<_> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        let first_col_powers_main_group: Vec<_> = self.srs.powers_main_group.iter().step_by(self.srs.dimention_x).cloned().collect();
        let q_y_commitment = msm(
            &coefficients_y, &first_col_powers_main_group
        ).expect("`points` is sliced by `cs`'s length");


        
        let coefficients_x_y: Vec<_> = bp.flatten_out()
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        let q_xy_commitment = msm(
            &coefficients_x_y,
            &self.srs.powers_main_group[..coefficients_x_y.len()],
        )
        .expect("`points` is sliced by `cs`'s length");
        (q_xy_commitment,q_y_commitment)
    }

    //not compeleted , should return 2 commitment, one for q_xy another for q_y
    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p: &BivariatePolynomial<FieldElement<F>>,
    ) -> (Self::Commitment,Self::Commitment) {
        // let mut poly_to_commit = p - y;
        let poly_to_commit = p.sub_by_field_element(evaluation);
        let (q_xy, q_y) = poly_to_commit.ruffini_division(x,y);
        // commitment to q_y , I should change the SRS to be compatible with it 
        self.commit(&q_xy,&q_y)
    }

    // should accept 2 commitment instead of 1
    fn verify(
        &self,
        x: &FieldElement<F>, // x random point 
        y: &FieldElement<F>, // y random point 
        evaluation: &FieldElement<F>, // F(x,y) = evaluation
        p_commitment: &Self::Commitment,
        proofs: &(&Self::Commitment,&Self::Commitment),
    ) -> bool {

        let g1 = &self.srs.powers_main_group[0];
        let g2 = &self.srs.powers_secondary_group[0];
        let tau_g2 = &self.srs.powers_secondary_group[1];
        let tetha_g2 = &self.srs.powers_secondary_group[2];



        let e = P::compute_batch(&[
            (
                &p_commitment.operate_with(&(g1.operate_with_self(evaluation.representative())).neg()),
                g2,
            ),
            (
                &proofs.0.neg(),
                &(tau_g2.operate_with(&(g2.operate_with_self(x.representative())).neg())),
            ),  
            (
                &proofs.1.neg(),
                &(tetha_g2.operate_with(&(g2.operate_with_self(y.representative())).neg())),
            ),
        ]);
        e == Ok(FieldElement::one())
    }

}


// always first row and second column -> (m,n)
/// Generate SRS for a tau and tetha 
pub fn generate_srs(dims: (usize,usize), taus: (FrElement,FrElement)) -> Vec<Vec<G1Point>> {
    // Generate powers of tau: tau^1, tau^2, ..., tau^n
    let powers_of_tau_theta = vandemonde_challenge(&taus.0, &taus.1, dims.0, dims.1);

    let g1: ShortWeierstrassProjectivePoint<BLS12381Curve>   = <BLS12381Curve as IsEllipticCurve>::generator();
    let mut two_dim_tau_g1: Vec<Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>>> = Vec::with_capacity(dims.0);
   
    for i in 0..dims.0 {
        let mut tau_g1 = vec![g1.clone(); dims.1];
        tau_g1
            .par_iter_mut()
            .zip(&powers_of_tau_theta[i])
            .for_each(|(g1, tau_i)| {
                *g1 = g1.operate_with_self(tau_i.representative());
            });
        two_dim_tau_g1.push(tau_g1);
    }

    two_dim_tau_g1

}

/// Computes the powers of tau and theta: [tau^0 , tau^1 , ... ,  tau^n-1 ] * theta^0
///                               , ... , [tau^0 , tau^1 , ... ,  tau^n-1 ] * theta^m
/// 
fn vandemonde_challenge(tau: &FrElement,theta: &FrElement , row_len: usize,col_len: usize) -> Vec<Vec<FrElement>> {

    let mut vec: Vec<Vec<FrElement>> = Vec::with_capacity(row_len);

    for _ in 0..row_len {
        vec.push(Vec::with_capacity(col_len));
    }

    for i in 0..row_len {
        let y_row = theta.pow(i);

        let mut row: Vec<FrElement> = Vec::with_capacity(row_len);
        for j in 0..col_len{
            row.push(y_row.clone().mul( tau.pow(j)));// TODO check for not being wrong.
        }
        vec[i]=row;
    }
    vec
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_vandemonde_challenge() {
        let challenge = vandemonde_challenge(&FrElement::from(2),&FrElement::from(3), 2,3);

        assert_eq!(
            challenge,
            vec![
                vec![FrElement::from(1),FrElement::from(2),FrElement::from(4)],
                vec![FrElement::from(3),FrElement::from(6),FrElement::from(12)]
            ]
        );
    }
}
