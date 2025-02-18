
use derive_more::From;


pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, From)]
pub enum Error {
    #[from]
    Custom(String),

    #[from] 
    WasmerRuntime(wasmer::RuntimeError),

    #[from]
    WasmerWasiRuntime(wasmer_wasix::WasiRuntimeError),

    #[from]
    WasmerCompile(wasmer::IoCompileError),
   
    #[from]
    wasmerExport(wasmer::ExportError),
    // -- Externals
    #[from]
    Io(std::io::Error), // as example
    #[from]
    SerderJson(serde_json::Error),

    #[from]
    FFTError(lambdaworks_math::fft::errors::FFTError),
    #[from]
    MSMError(lambdaworks_math::msm::naive::MSMError),

    #[from]
    LambdaConversion(lambdaworks_math::errors::ByteConversionError),

    #[from]
    LambdaHexConversion(lambdaworks_math::errors::CreationError),
    #[from]
    WaserInstansiation(wasmer::InstantiationError),



}

// region:    --- Custom

impl Error {
    pub fn custom(val: impl std::fmt::Display) -> Self {
        Self::Custom(val.to_string())
    }
}

impl From<&str> for Error {
    fn from(val: &str) -> Self {
        Self::Custom(val.to_string())
    }
}

// endregion: --- Custom

// region:    --- Error Boilerplate

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for Error {}

// endregion: --- Error Boilerplate

















// pub type Error = Box<dyn std::error::Error>;
// use core::{fmt::Display};

// use lambdaworks_math::errors::CreationError;

// // Define a newtype wrapper for `CreationError`
// #[derive(Debug)]
// pub struct MyCreationError(CreationError);

// // Implement `From<CreationError>` for your newtype
// impl From<CreationError> for MyCreationError {
//     fn from(error: CreationError) -> Self {
//         MyCreationError(error)
//     }
// }

// impl Display for MyCreationError {
//     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//             write!(f, "dd")
//         }
// }

// impl std::error::Error for MyCreationError {}


// // Implement `From<MyCreationError>` for `Error`
// impl From<MyCreationError> for Error {
//     fn from(error: MyCreationError) -> Self {
//         todo!()
//     }
// }