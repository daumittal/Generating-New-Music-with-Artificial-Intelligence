use crate::logits::Logits;
use ort::{DynValue, SessionOutputs};

/// Represents the outputs generated by the MusicGen model.
pub struct MusicGenOutputs<'s> {
    /// The raw session outputs from ONNX Runtime.
    outputs: SessionOutputs<'s>,
}

impl<'s> MusicGenOutputs<'s> {
    /// Creates a new instance of `MusicGenOutputs` from the provided session outputs.
    pub fn new(outputs: SessionOutputs<'s>) -> Self {
        Self { outputs }
    }

    /// Extracts and converts the logits from the outputs into a `Logits` object.
    ///
    /// # Returns
    /// - A `Logits` object if successful.
    ///
    /// # Errors
    /// - Returns an error if the logits cannot be extracted or converted.
    pub fn take_logits(&mut self) -> ort::Result<Logits> {
        let logits = self.outputs.remove("logits").expect("Logits not found in outputs");
        Logits::from_3d_dyn_value(&logits)
    }

    /// Retrieves the decoder key for the specified layer index (`i`) from the outputs.
    ///
    /// # Arguments
    /// - `i`: The layer index for which to retrieve the decoder key.
    ///
    /// # Panics
    /// - If the key has already been taken or does not exist in the outputs.
    pub fn take_present_decoder_key(&mut self, i: usize) -> DynValue {
        let key = format!("present.{i}.decoder.key");
        self.outputs
            .remove(key.as_str())
            .unwrap_or_else(|| panic!("{key} was already taken or does not exist"))
    }

    /// Retrieves the decoder value for the specified layer index (`i`) from the outputs.
    ///
    /// # Arguments
    /// - `i`: The layer index for which to retrieve the decoder value.
    ///
    /// # Panics
    /// - If the value has already been taken or does not exist in the outputs.
    pub fn take_present_decoder_value(&mut self, i: usize) -> DynValue {
        let value = format!("present.{i}.decoder.value");
        self.outputs
            .remove(value.as_str())
            .unwrap_or_else(|| panic!("{value} was already taken or does not exist"))
    }

    /// Retrieves the encoder key for the specified layer index (`i`) from the outputs.
    ///
    /// # Arguments
    /// - `i`: The layer index for which to retrieve the encoder key.
    ///
    /// # Panics
    /// - If the key has already been taken or does not exist in the outputs.
    pub fn take_present_encoder_key(&mut self, i: usize) -> DynValue {
        let key = format!("present.{i}.encoder.key");
        self.outputs
            .remove(key.as_str())
            .unwrap_or_else(|| panic!("{key} was already taken or does not exist"))
    }

    /// Retrieves the encoder value for the specified layer index (`i`) from the outputs.
    ///
    /// # Arguments
    /// - `i`: The layer index for which to retrieve the encoder value.
    ///
    /// # Panics
    /// - If the value has already been taken or does not exist in the outputs.
    pub fn take_present_encoder_value(&mut self, i: usize) -> DynValue {
        let value = format!("present.{i}.encoder.value");
        self.outputs
            .remove(value.as_str())
            .unwrap_or_else(|| panic!("{value} was already taken or does not exist"))
    }
}