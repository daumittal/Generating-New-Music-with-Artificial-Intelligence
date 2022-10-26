use std::collections::VecDeque;

use half::f16;
use ndarray::{Array, Axis};
use ort::{self, inputs, DynValue, Error as OrtError, Result as OrtResult};

pub struct MusicGenAudioEncodec {
    pub audio_encodec_decode: ort::Session,
}

impl MusicGenAudioEncodec {
    pub fn encode(&self, tokens: impl IntoIterator<Item = [i64; 4]>) -> OrtResult<VecDeque<f32>> {
        // Flatten the token chunks into a single vector of i64s
        let flattened: Vec<i64> = tokens.into_iter().flat_map(|chunk| chunk).collect();

        if flattened.is_empty() {
            return Ok(VecDeque::new());
        }

        // Determine the number of time steps (sequence length)
        let sequence_len = flattened.len() / 4;
        if flattened.len() % 4 != 0 {
            return Err(OrtError::CustomError(format!(
                "Expected input length divisible by 4, got {}",
                flattened.len()
            )));
        }

        // Shape: (sequence_len, 4) → Transposed → [1, 1, 4, sequence_len]
        let token_array = Array::from_shape_vec((sequence_len, 4), flattened)
            .map_err(|_| OrtError::CustomError("Failed to reshape token input".into()))?;
        let model_input = token_array
            .t()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));

        // Run the ONNX model
        let mut outputs = self.audio_encodec_decode.run(inputs![model_input]?)?;

        // Extract the 'audio_values' output tensor
        let output = outputs
            .remove("audio_values")
            .ok_or_else(|| OrtError::CustomError("Missing 'audio_values' in model output".into()))?;

        // Try extracting raw f32 data, fallback to f16 if needed
        if let Ok((_, values)) = output.try_extract_raw_tensor::<f32>() {
            Ok(values.iter().copied().collect())
        } else if let Ok((_, values)) = output.try_extract_raw_tensor::<f16>() {
            Ok(values.iter().map(|v| f32::from(*v)).collect())
        } else {
            Err(OrtError::CustomError(
                "Expected 'audio_values' tensor of type f32 or f16".into(),
            ))
        }
    }
}
