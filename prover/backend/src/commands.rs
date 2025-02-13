use clap::{Args, Parser, Subcommand};
use wasmer_wasix::types::wasi::ErrnoSignal;

use crate::{error::Error, permutaion::SubcircuitLibraryInfo, wasm::Wasm};
use std::fs;
use crate::permutaion::PlacementInstance;
use serde_json::{from_str, Value};

#[derive(Parser, Debug)]
pub struct BackendArgs {
    #[clap(subcommand)]
    pub entity: BackendEntity,
}

#[derive(Subcommand, Debug)]
pub enum BackendEntity {
    #[clap(about = "Generate Proof With Placements and Permutations")]
    GenerateProof(GenerateProofArgs),
    #[clap(about = "Verify a merkle proof")]
    VerifyProof(VerifyArgs),
}


#[derive(Args, Debug)]
pub struct GenerateProofArgs {
    pub placement_path: String,
    pub permutation_path: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub proof_path: String,

}

pub fn generate_proof(wasm :& mut Wasm ,args: GenerateProofArgs, subcircuit_lib :&SubcircuitLibraryInfo) -> Result<(),Error> {
    let content = fs::read_to_string(args.placement_path)?;

    let placements : Vec<PlacementInstance> = from_str(&content)?;
    let witnesses = wasm.calculate_witness(&placements)?;


    todo!()
}

pub fn verify_proof(args: VerifyArgs) -> Result<(), Error> {
    todo!()
}
