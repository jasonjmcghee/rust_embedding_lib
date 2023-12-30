use candle::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use lazy_static::lazy_static;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use tokenizers::{PaddingParams, Tokenizer};

lazy_static! {
    static ref MODEL: Mutex<Option<(BertModel, Tokenizer)>> = Mutex::new(None);
}

// Function to initialize the model and tokenizer from local files
#[no_mangle]
pub extern "C" fn init_model(
    config_path_raw: *const c_char,
    tokenizer_path_raw: *const c_char,
    weights_path_raw: *const c_char,
    approximate_gelu: bool,
) -> bool {
    let config_path = unsafe { CStr::from_ptr(config_path_raw) }.to_str().unwrap();
    let tokenizer_path = unsafe { CStr::from_ptr(tokenizer_path_raw) }
        .to_str()
        .unwrap();
    let weights_path = unsafe { CStr::from_ptr(weights_path_raw) }
        .to_str()
        .unwrap();

    let device = candle::Device::Cpu;

    // Load config
    let config_contents = std::fs::read_to_string(config_path).unwrap();
    let mut config: Config = serde_json::from_str(&config_contents).unwrap();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Load weights
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device).unwrap() };

    if approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }

    let model = BertModel::load(vb, &config).unwrap();

    // Store model and tokenizer in the global MODEL variable
    let mut model_guard = MODEL.lock().unwrap();
    *model_guard = Some((model, tokenizer));
    true
}

#[repr(C)]
pub struct EmbeddingResult {
    embeddings: *const f32,
    len: usize,
    error: *const c_char,
}

// Function to generate embeddings
#[no_mangle]
pub extern "C" fn generate_embeddings(text: *const c_char) -> EmbeddingResult {
    let text = unsafe { CStr::from_ptr(text).to_str().unwrap() };

    let model_guard = MODEL.lock().unwrap();
    let (model, tokenizer) = match model_guard.as_ref() {
        Some(data) => data,
        None => {
            return EmbeddingResult {
                embeddings: std::ptr::null(),
                len: 0,
                error: CString::new("Model not initialized").unwrap().into_raw(),
            }
        }
    };

    // Create a new tokenizer instance with the desired configuration
    let mut new_tokenizer = tokenizer.clone();
    new_tokenizer.with_padding(Some(PaddingParams::default()));

    if let Err(e) = new_tokenizer.with_truncation(None) {
        return EmbeddingResult {
            embeddings: std::ptr::null(),
            len: 0,
            error: CString::new(e.to_string()).unwrap().into_raw(),
        };
    }

    let tokens = match tokenizer.encode(text, true) {
        Ok(t) => t,
        Err(e) => {
            return EmbeddingResult {
                embeddings: std::ptr::null(),
                len: 0,
                error: CString::new(e.to_string()).unwrap().into_raw(),
            }
        }
    };

    let token_ids = match Tensor::new(&tokens.get_ids().to_vec()[..], &model.device)
        .unwrap()
        .unsqueeze(0)
    {
        Ok(t) => t,
        Err(e) => {
            return EmbeddingResult {
                embeddings: std::ptr::null(),
                len: 0,
                error: CString::new(e.to_string()).unwrap().into_raw(),
            }
        }
    };

    let token_type_ids = token_ids.zeros_like().unwrap();

    let embeddings = match model.forward(&token_ids, &token_type_ids) {
        Ok(e) => e,
        Err(e) => {
            return EmbeddingResult {
                embeddings: std::ptr::null(),
                len: 0,
                error: CString::new(e.to_string()).unwrap().into_raw(),
            }
        }
    };

    // Flatten the tensor without changing the total number of elements
    let reshaped_embeddings = match embeddings.reshape(&[embeddings.elem_count()]) {
        Ok(r) => r,
        Err(e) => {
            return EmbeddingResult {
                embeddings: std::ptr::null(),
                len: 0,
                error: CString::new(e.to_string()).unwrap().into_raw(),
            }
        }
    };

    let elem_count = reshaped_embeddings.elem_count();

    EmbeddingResult {
        embeddings: reshaped_embeddings.to_vec1::<f32>().unwrap().as_ptr(),
        len: elem_count,
        error: std::ptr::null(),
    }
}

// Function to free the resources allocated by `generate_embeddings`
#[no_mangle]
pub extern "C" fn free_embeddings(result: EmbeddingResult) {
    unsafe {
        // If there are embeddings, reconstruct the Vec from the raw parts so Rust can deallocate it
        if !result.embeddings.is_null() {
            // This turns the raw pointer back into a Vec which gets dropped at the end of the scope
            // This effectively frees the memory of the Vec
            Vec::from_raw_parts(result.embeddings as *mut f32, result.len, result.len);
        }

        // If there's an error message, convert it back to a CString to deallocate it
        if !result.error.is_null() {
            // Convert the raw error string back into a CString
            // The CString's destructor will free the memory when it goes out of scope
            let _ = CString::from_raw(result.error as *mut c_char);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embeddings() {
        let config_path_c_str = CString::new("models/gte-small/config.json").unwrap();
        let config_path = config_path_c_str.as_ptr() as *const c_char;

        let tokenizer_path_c_str = CString::new("models/gte-small/tokenizer.json").unwrap();
        let tokenizer_path = tokenizer_path_c_str.as_ptr() as *const c_char;

        let weights_path_c_str = CString::new("models/gte-small/model.safetensors").unwrap();
        let weights_path = weights_path_c_str.as_ptr() as *const c_char;

        // Initialize the model first
        init_model(config_path, tokenizer_path, weights_path, false);

        // Test embedding generation
        let text = "Test sentence for embeddings.";
        let c_str = CString::new(text).unwrap();
        let chars: *const c_char = c_str.as_ptr() as *const c_char;
        let result: EmbeddingResult = generate_embeddings(chars);
        assert_eq!(49152, result.len);
    }
}
