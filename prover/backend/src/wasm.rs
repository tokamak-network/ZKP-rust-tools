use lambdaworks_groth16::common::FrElement;
use wasmer::{imports, wasmparser::Export, Value, Exports, Function, Instance, Memory, MemoryType, Module, RuntimeError, Store};

use crate::permutaion::PlacementInstance;
use std;
use wasmer_wasix::WasiEnv;

// use thiserror::Error;


// pub type Result<T> = core::result::Result<T,Error>;
pub type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
pub struct Wasm {
    pub subcircuits_insts :Vec<SubcircuitWasmInst>,
}

#[derive(Debug)]
pub struct SubcircuitWasmInst {
    pub exports: Exports, 
    pub memory: Memory,
}

impl SubcircuitWasmInst {
    fn func(&self, name: &str) -> &Function {
        self.exports
            .get_function(name)
            .unwrap_or_else(|_| panic!("function {} not found", name))
    }

    fn init(&self, store: &mut Store, sanity_check: bool) -> Result<(),Error> {
        let func = self.func("init");
        func.call(store, &[Value::I32(sanity_check as i32)])?;
        Ok(())
    }


}



impl Wasm {
    pub fn new(subcircuit_cnt :usize ,dir: impl AsRef<std::path::Path>) -> Result<Self,Error> {
        let mut store = Store::default();
        let mut wasm = Wasm { subcircuits_insts: Vec::with_capacity(subcircuit_cnt) };
        

        for entry in std::fs::read_dir(dir)?{
            // let entry_path = entry.unwrap().path();
            let entry = entry?;
            // TODO check for order of subcircuits .

            let module = Module::from_file(&store, entry.path())?;
            let (export, memory) = Wasm::make_wasm_runtime(&mut store, module)?;


            wasm.subcircuits_insts.push(SubcircuitWasmInst{exports: export, memory: memory});
        }

        Ok(wasm)
    }

    pub fn make_wasm_runtime(store: &mut Store, module: Module) -> Result<(Exports, Memory),Error> {
        let memory = Memory::new(store, MemoryType::new(2000, None, false)).unwrap();
        let import_object = imports! {
            "env" => {
                "memory" => memory.clone(),
            },
            // Host function callbacks from the WASM
            "runtime" => {
                "error" => runtime::error(store),
                "logSetSignal" => runtime::log_signal(store),
                "logGetSignal" => runtime::log_signal(store),
                "logFinishComponent" => runtime::log_component(store),
                "logStartComponent" => runtime::log_component(store),
                "log" => runtime::log_component(store),
                "exceptionHandler" => runtime::exception_handler(store),
                "showSharedRWMemory" => runtime::show_memory(store),
                "printErrorMessage" => runtime::print_error_message(store),
                "writeBufferMessage" => runtime::write_buffer_message(store),
            }
        };
        let instance = Instance::new(store, &module, &import_object)?;
        let exports = instance.exports.clone();
        let mut wasi_env = WasiEnv::builder("calculateWitness").finalize(store)?;
        wasi_env.initialize_with_memory(store, instance, Some(memory.clone()), false)?;
        Ok((exports,memory))
    }



    pub fn calculate_witness(&mut self,placements :&[PlacementInstance]) -> Result<Vec<FrElement> ,Error >
    {   
        for placement in placements {

            // self.subcircuits_insts[placement.subcircuit_id as usize].init(store, sanity_check)
        }

        todo!()
    }
}


// Error type to signal end of execution.
// From https://docs.wasmer.io/integrations/examples/exit-early
#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("{0}")]
struct ExitCode(u32);


// callback hooks for debugging
mod runtime {
    use super::*;

    pub fn error(store: &mut Store) -> Function {
        #[allow(unused)]
        #[allow(clippy::many_single_char_names)]
        fn func(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> Result<(), RuntimeError> {
            // NOTE: We can also get more information why it is failing, see p2str etc here:
            // https://github.com/iden3/circom_runtime/blob/master/js/witness_calculator.js#L52-L64
            println!("runtime error, exiting early: {a} {b} {c} {d} {e} {f}",);
            Err(RuntimeError::user(Box::new(ExitCode(1))))
        }
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn exception_handler(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32) {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn show_memory(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn print_error_message(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    // Circom 2.0
    pub fn write_buffer_message(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func() {}
        Function::new_typed(store, func)
    }

    pub fn log_signal(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32, b: i32) {}
        Function::new_typed(store, func)
    }

    pub fn log_component(store: &mut Store) -> Function {
        #[allow(unused)]
        fn func(a: i32) {}
        Function::new_typed(store, func)
    }
}
