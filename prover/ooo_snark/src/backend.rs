


use lambdaworks_groth16::{common::{FrElement,FrField}, r1cs::{self, ConstraintSystem, R1CS}};


#[derive(PartialEq, Clone, Debug)]
pub struct BeckendProver {
    // need to ask about the inputs of different phases 
    pub subcircuits: Vec<R1CS>,
    // pub 



}

impl BeckendProver {
    // function for set witnesses and calculate arithmatic proof 
    pub fn Calculate_Arithmatic_Proof(ordered_idx: &[usize], witnesses: &[Vec<FrElement>]) {}


    // function for set B permutation and create copy constraint proofs 
    pub fn Calculate_Permutation_Proof(connected_wires :Vec<Vec<(usize,usize)>>) {}

    // function to generate inner product using both B and also witnesses.
    // pub 

}



