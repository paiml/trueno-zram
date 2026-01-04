//! Configuration parsing for zram devices.

use serde::Deserialize;
use std::path::Path;

/// Configuration for a zram device.
#[derive(Debug, Clone, Deserialize)]
pub struct ZramConfig {
    /// Device index (0 = /dev/zram0).
    #[serde(default)]
    pub device: u32,

    /// Size in bytes (or with suffix: K, M, G).
    pub size: String,

    /// Compression algorithm.
    #[serde(default = "default_algorithm")]
    pub algorithm: String,

    /// Number of compression streams.
    #[serde(default = "default_streams")]
    pub streams: u32,

    /// Swap priority.
    #[serde(default = "default_priority")]
    pub priority: i32,
}

fn default_algorithm() -> String {
    "lz4".to_string()
}

fn default_streams() -> u32 {
    0 // 0 means auto-detect based on CPU count
}

fn default_priority() -> i32 {
    100
}

/// Root configuration structure.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// List of zram devices to configure.
    #[serde(default)]
    pub devices: Vec<ZramConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            devices: vec![ZramConfig {
                device: 0,
                size: "ram/2".to_string(),
                algorithm: default_algorithm(),
                streams: default_streams(),
                priority: default_priority(),
            }],
        }
    }
}

/// Load configuration from standard locations.
pub fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let config_paths = [
        "/etc/trueno-zram.conf",
        "/etc/trueno-zram.conf.d/",
        "/usr/lib/trueno-zram.conf.d/",
    ];

    for path_str in &config_paths {
        let path = Path::new(path_str);
        if path.exists()
            && path.is_file() {
                let content = std::fs::read_to_string(path)?;
                return Ok(toml::from_str(&content)?);
            }
            // Handle .d directories later
    }

    // Return default configuration if no config found
    Ok(Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.devices.len(), 1);
        assert_eq!(config.devices[0].algorithm, "lz4");
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
            [[devices]]
            device = 0
            size = "4G"
            algorithm = "zstd"
            streams = 4
            priority = 100
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.devices.len(), 1);
        assert_eq!(config.devices[0].size, "4G");
        assert_eq!(config.devices[0].algorithm, "zstd");
    }
}
