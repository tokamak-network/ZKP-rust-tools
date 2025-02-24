mod permutaion; 
mod commands;
mod wasm;
mod fnv;
mod error;
mod qap;
mod setup; 

use clap::{Args, Parser, Subcommand};
use commands::{generate_proof, verify_proof, BackendArgs, BackendEntity};
use error::Error;
use wasm::Wasm;
use permutaion::SubcircuitLibraryInfo;






fn main() {

    let mut wasm = Wasm::new(30, "./subcircuits/wasm").expect("failed to read wasm files");

    let subcircuit_info_lib = SubcircuitLibraryInfo::new("./synthesizer/globalwire.json" , "./synthesizer/subcircuitInfo.json",8192, "./subcircuits/r1cs").expect("failed to read synthesizer initialize values");


    let args = BackendArgs::parse();
    if let Err(e) = match args.entity {
        BackendEntity::GenerateProof(args) => generate_proof(& mut wasm,args, &subcircuit_info_lib),
        BackendEntity::VerifyProof(args) => verify_proof(args),
    } {
        println!("Error while running command: {:?}", e);
    }


  
}

