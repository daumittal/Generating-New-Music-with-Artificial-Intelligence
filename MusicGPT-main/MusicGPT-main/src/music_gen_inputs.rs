use std::collections::HashMap;
use ort::{DynValue, Tensor};

pub struct MusicGenInputs {
    inputs: HashMap<String, DynValue>,
    pub use_cache_branch: bool,
}

impl MusicGenInputs {
    /// Creates a new instance of `MusicGenInputs` with an empty input map.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            use_cache_branch: false,
        }
    }

    /// Sets the `encoder_attention_mask` input value.
    pub fn encoder_attention_mask<T, E>(&mut self, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            "encoder_attention_mask".to_string(),
            v.try_into()?,
        );
        Ok(())
    }

    /// Sets the `input_ids` input value.
    pub fn input_ids<T, E>(&mut self, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert("input_ids".to_string(), v.try_into()?);
        Ok(())
    }

    /// Sets the `encoder_hidden_states` input value.
    pub fn encoder_hidden_states<T, E>(&mut self, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            "encoder_hidden_states".to_string(),
            v.try_into()?,
        );
        Ok(())
    }

    /// Removes the `encoder_hidden_states` input from the map.
    pub fn remove_encoder_hidden_states(&mut self) {
        self.inputs.remove("encoder_hidden_states");
    }

    /// Sets the `past_key_values.{i}.decoder.key` input value.
    pub fn past_key_value_decoder_key<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            format!("past_key_values.{i}.decoder.key"),
            v.try_into()?,
        );
        Ok(())
    }

    /// Sets the `past_key_values.{i}.decoder.value` input value.
    pub fn past_key_value_decoder_value<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            format!("past_key_values.{i}.decoder.value"),
            v.try_into()?,
        );
        Ok(())
    }

    /// Sets the `past_key_values.{i}.encoder.key` input value.
    pub fn past_key_value_encoder_key<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            format!("past_key_values.{i}.encoder.key"),
            v.try_into()?,
        );
        Ok(())
    }

    /// Sets the `past_key_values.{i}.encoder.value` input value.
    pub fn past_key_value_encoder_value<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert(
            format!("past_key_values.{i}.encoder.value"),
            v.try_into()?,
        );
        Ok(())
    }

    /// Updates the `use_cache_branch` flag and sets the corresponding input value.
    pub fn use_cache_branch(&mut self, value: bool) {
        self.use_cache_branch = value;
        let tensor = Tensor::from_array(([1], vec![value])).unwrap().into_dyn();
        self.inputs.insert("use_cache_branch".to_string(), tensor);
    }

    /// Converts the input map into an `ort::SessionInputs` object.
    pub fn ort(&self) -> ort::SessionInputs {
        ort::SessionInputs::ValueMap(
            self.inputs
                .iter()
                .map(|(key, value)| (key.clone().into(), value.view().into()))
                .collect::<Vec<_>>(),
        )
    }
}