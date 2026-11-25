//! HTTP client configuration for fetching datasets and CLI documents.

use anno::{Error, Result};
use std::{path::Path, time::Duration};
use ureq::tls::{parse_pem, PemItem, RootCerts, TlsConfig};

/// Use ureq's bundled roots unless the caller explicitly supplies a CA bundle.
pub fn agent(timeout: Duration) -> Result<ureq::Agent> {
    let bundle = std::env::var_os("SSL_CERT_FILE");
    http_agent(bundle.as_deref().map(Path::new), timeout).map_err(Error::InvalidInput)
}

fn http_agent(
    ca_file: Option<&Path>,
    timeout: Duration,
) -> std::result::Result<ureq::Agent, String> {
    let mut config = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false);
    if let Some(path) = ca_file {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Cannot read SSL_CERT_FILE {}: {e}", path.display()))?;
        let mut roots = Vec::new();
        for item in parse_pem(&bytes) {
            match item.map_err(|e| format!("Invalid SSL_CERT_FILE {}: {e}", path.display()))? {
                PemItem::Certificate(cert) => roots.push(cert),
                _ => {
                    return Err(format!(
                        "Expected certificates in SSL_CERT_FILE {}",
                        path.display()
                    ))
                }
            }
        }
        if roots.is_empty() {
            return Err(format!(
                "No certificates in SSL_CERT_FILE {}",
                path.display()
            ));
        }
        config = config.tls_config(
            TlsConfig::builder()
                .root_certs(RootCerts::from(roots))
                .build(),
        );
    }
    Ok(config.build().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicitly_configured_ca_file_must_provide_certificates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let timeout = Duration::from_secs(1);
        assert!(http_agent(None, timeout).is_ok());
        assert!(
            http_agent(Some(&root.join("missing-ca-bundle.pem")), timeout)
                .unwrap_err()
                .contains("Cannot read SSL_CERT_FILE")
        );
        assert!(http_agent(Some(&root.join("Cargo.toml")), timeout)
            .unwrap_err()
            .contains("No certificates in SSL_CERT_FILE"));
    }
}
