use std::collections::VecDeque;
use std::io::{BufWriter, Cursor};
use std::time::Duration;

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ChannelCount, SampleFormat, SampleRate, Stream, SupportedBufferSize, SupportedStreamConfig};
use hound;

const DEFAULT_RATE: u32 = 32000;

pub struct AudioManager {
    host: cpal::Host,
    sample_format: SampleFormat,
    sample_rate: u32,
    channel_count: u16,
}

pub struct AudioStream {
    pub stream: Stream,
    pub duration: Duration,
}

unsafe impl Send for AudioStream {}
unsafe impl Sync for AudioStream {}

impl Default for AudioManager {
    fn default() -> Self {
        Self {
            host: cpal::default_host(),
            sample_format: SampleFormat::F32,
            sample_rate: DEFAULT_RATE,
            channel_count: 1,
        }
    }
}

impl AudioManager {
    pub fn play_queue(&self, mut samples: VecDeque<f32>) -> Result<AudioStream> {
        let duration_ms = (1000 * samples.len()) / self.sample_rate as usize;
        let config = SupportedStreamConfig::new(
            ChannelCount::from(self.channel_count),
            SampleRate(self.sample_rate),
            SupportedBufferSize::Unknown,
            self.sample_format,
        );

        let device = self.host.default_output_device()
            .ok_or_else(|| anyhow!("No output device found"))?;

        let stream = device.build_output_stream(
            &config.into(),
            move |output: &mut [f32], _| {
                for frame in output.chunks_mut(self.channel_count as usize) {
                    for sample in frame.iter_mut() {
                        *sample = samples.pop_front().unwrap_or(0.0);
                    }
                }
            },
            |_err| {},
            None,
        )?;

        stream.play()?;

        Ok(AudioStream {
            stream,
            duration: Duration::from_millis(duration_ms as u64),
        })
    }

    pub fn serialize_wav(&self, samples: VecDeque<f32>) -> hound::Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: self.channel_count,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample(),
            sample_format: self.hound_format(),
        };

        let mut buffer = vec![];
        {
            let writer = Cursor::new(&mut buffer);
            let mut wav_writer = hound::WavWriter::new(BufWriter::new(writer), spec)?;
            for sample in samples {
                wav_writer.write_sample(sample)?;
            }
            // Ensures proper finalization of WAV file
            wav_writer.finalize()?;
        }

        Ok(buffer)
    }

    fn bits_per_sample(&self) -> u16 {
        match self.sample_format {
            SampleFormat::F32 | SampleFormat::I32 | SampleFormat::U32 => 32,
            SampleFormat::I16 | SampleFormat::U16 => 16,
            SampleFormat::I8 | SampleFormat::U8 => 8,
            SampleFormat::F64 | SampleFormat::I64 | SampleFormat::U64 => 64,
        }
    }

    fn hound_format(&self) -> hound::SampleFormat {
        match self.sample_format {
            SampleFormat::F32 | SampleFormat::F64 => hound::SampleFormat::Float,
            _ => hound::SampleFormat::Int,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_to_wav() -> Result<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.wav");
        let manager = AudioManager::default();
        let reader = hound::WavReader::open(path)?;
        let samples: VecDeque<f32> = reader
            .into_samples::<f32>()
            .map(Result::unwrap)
            .collect();

        let generated_wav = manager.serialize_wav(samples)?;
        let expected = std::fs::read(path)?;

        assert_eq!(generated_wav, expected);
        Ok(())
    }
}
