use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

use crate::bipolynomial::BivariatePolynomial;
use lambdaworks_math::polynomial::Polynomial as UnivariatePolynomial;



pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;


    fn commit(&self, bp: &BivariatePolynomial<FieldElement<F>>,p: &UnivariatePolynomial<FieldElement<F>>) -> (Self::Commitment,Self::Commitment);



    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p: &BivariatePolynomial<FieldElement<F>>,
    ) -> (Self::Commitment,Self::Commitment);
    // fn open_batch(
    //     &self,
    //     x: &FieldElement<F>,
    //     y: &[FieldElement<F>],
    //     p: &[Polynomial<FieldElement<F>>],
    //     upsilon: &FieldElement<F>,
    // ) -> Self::Commitment;

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proofs: &(&Self::Commitment,&Self::Commitment),
    ) -> bool;

    // fn verify_batch(
    //     &self,
    //     x: &FieldElement<F>,
    //     ys: &[FieldElement<F>],
    //     p_commitments: &[Self::Commitment],
    //     proof: &Self::Commitment,
    //     upsilon: &FieldElement<F>,
    // ) -> bool;
}
