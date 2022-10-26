use std::{error, path::PathBuf};

use axum::http::StatusCode;
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::storage::{AppFs, Storage};

impl AppFs {
    /// Downloads a file from the specified URL into the local storage, if it doesn’t already exist or if forced.
    ///
    /// # Arguments
    /// * `url` - URL of the remote file to download.
    /// * `local_file` - Relative path in local storage where the file should be saved.
    /// * `force` - If true, always download even if the file exists.
    /// * `cbk` - Callback invoked with (downloaded_bytes, total_bytes) during progress.
    pub async fn fetch_remote_data_file<Cb: Fn(usize, usize)>(
        &self,
        url: &str,
        local_file: &str,
        force: bool,
        cbk: Cb,
    ) -> std::io::Result<PathBuf> {
        if self.exists(local_file).await? && !force {
            return Ok(self.path_buf(local_file));
        }

        let response = reqwest::get(url).await.map_err(io_err)?;
        let status = response.status();

        if status != StatusCode::OK {
            return Err(io_err(format!("Unexpected HTTP status: {status}")));
        }

        let total_bytes = response.content_length().unwrap_or(0) as usize;
        let temp_file = format!("{local_file}.temp");
        let mut temp = self.create(&temp_file).await?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(io_err)?;
            downloaded += chunk.len();
            cbk(downloaded, total_bytes);
            temp.write_all(&chunk).await?;
        }

        self.mv(&temp_file, local_file).await?;
        Ok(self.path_buf(local_file))
    }
}

fn io_err<E>(e: E) -> std::io::Error
where
    E: Into<Box<dyn error::Error + Send + Sync>>,
{
    std::io::Error::new(std::io::ErrorKind::Other, e)
}
