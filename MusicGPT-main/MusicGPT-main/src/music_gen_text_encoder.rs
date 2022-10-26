use tokenizers::Tokenizer;
use ort::{DynValue, Session, Tensor};
use crate::tensor_ops::ones_tensor;

/// Represents a text encoder for MusicGen.
pub struct MusicGenTextEncoder {
    /// Tokenizer used to preprocess input text.
    pub tokenizer: Tokenizer,
    /// ONNX Runtime session for the text encoder model.
    pub text_encoder: Session,
}

impl MusicGenTextEncoder {
    /// Encodes the given text into embeddings using the text encoder.
    ///
    /// # Arguments
    /// - `text`: The input text to encode.
    ///
    /// # Returns
    /// A tuple containing:
    /// - The last hidden state from the text encoder as a dynamic tensor (`ort::DynValue`).
    /// - An attention mask as a dynamic tensor (`ort::DynValue`).
    ///
    /// # Errors
    /// Returns an error if tokenization, tensor creation, or model inference fails.
    pub fn encode(&self, text: &str) -> ort::Result<(DynValue, DynValue)> {
        // Tokenize the input text and convert tokens to a vector of integers.
        let tokens = self.tokenizer
            .encode(text, true)
            .expect("Error tokenizing text")
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();

        let tokens_len = tokens.len();

        // Create input tensors for the model.
        let input_ids = Tensor::from_array(([1, tokens_len], tokens))?;
        let attention_mask = ones_tensor::<i64>(&[1, tokens_len]);

        // Run the text encoder model with the input tensors.
        let mut output = self.text_encoder.run(ort::inputs![input_ids, attention_mask]?)?;

        // Extract the last hidden state from the model's output.
        let last_hidden_state = output
            .remove("last_hidden_state")
            .expect("last_hidden_state not found in output");

        // Return the last hidden state and the attention mask as dynamic tensors.
        Ok((last_hidden_state, ones_tensor::<i64>(&[1, tokens_len]).into_dyn()))
    }
}