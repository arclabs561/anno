//! URL resolution for fetching content from HTTP/HTTPS URLs.
//!
//! This lives in the CLI crate because network I/O is a CLI concern, not a
//! library concern. The library takes `&str`; the CLI resolves formats into text.

use std::collections::HashMap;
use std::path::Path;
use ureq::tls::{parse_pem, PemItem, RootCerts, TlsConfig};
use ureq::ResponseExt;

/// Use ureq's bundled roots unless the caller explicitly supplies a CA bundle.
fn http_agent(ca_file: Option<&Path>) -> Result<ureq::Agent, String> {
    let mut config = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(60)))
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

/// Resolved content from a URL.
#[derive(Debug, Clone)]
#[allow(dead_code)] // metadata/source_url used by downstream consumers
pub struct ResolvedContent {
    /// The extracted text content
    pub text: String,
    /// Grounding in the HTTP entity body before charset decoding or HTML extraction.
    pub source_map: anno::SourceMap,
    /// Metadata about the source (title, content-type, etc.)
    pub metadata: HashMap<String, String>,
    /// The final URL after redirects
    pub source_url: String,
}

/// Trait for URL resolvers that can fetch and extract text from URLs.
pub trait UrlResolver: std::fmt::Debug {
    /// Check if this resolver can handle the given URL.
    fn can_resolve(&self, url: &str) -> bool;

    /// Resolve the URL to text content.
    fn resolve(&self, url: &str) -> Result<ResolvedContent, String>;
}

/// HTTP/HTTPS URL resolver.
///
/// Fetches content from HTTP/HTTPS URLs and extracts text from HTML if needed.
#[derive(Debug, Default)]
pub struct HttpResolver {
    read_declared_language: bool,
}

impl HttpResolver {
    /// Create a new HTTP resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl UrlResolver for HttpResolver {
    fn can_resolve(&self, url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }

    fn resolve(&self, url: &str) -> Result<ResolvedContent, String> {
        let ca_file = std::env::var_os("SSL_CERT_FILE");
        let response = http_agent(ca_file.as_deref().map(Path::new))?
            .get(url)
            .call()
            .map_err(|e| {
                format!(
                    "Network error fetching {}: {}. \
                     Check your internet connection and try again.",
                    url, e
                )
            })?;

        if response.status() != 200 {
            return Err(format!(
                "HTTP {} fetching {}. \
                 Server returned error status. \
                 URL may be temporarily unavailable or changed.",
                response.status().as_u16(),
                url
            ));
        }

        let source_url = response.get_uri().to_string();
        let media_type = response
            .headers()
            .get(ureq::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()?.parse::<mime::Mime>().ok());
        let is_html = media_type.as_ref().is_some_and(|media| {
            matches!(media.essence_str(), "text/html" | "application/xhtml+xml")
        });
        let encoding = media_type
            .as_ref()
            .and_then(|media| media.get_param(mime::CHARSET))
            .and_then(|charset| encoding_rs::Encoding::for_label(charset.as_str().as_bytes()));
        // Keep ureq's optional charset feature disabled: the digest identifies
        // the decompressed HTTP entity body before any text decoding.
        let mut body = response.into_body();
        let bytes = body.read_to_vec().map_err(|e| {
            format!(
                "Failed to read response from {url}: {e}. Response may be too large or corrupted."
            )
        })?;
        let content = if let Some(encoding) = encoding {
            // decode honors a byte order mark before the HTTP charset.
            encoding.decode(&bytes).0.into_owned()
        } else if is_html {
            deformat::detect::decode_bytes(&bytes, "utf-8").into_owned()
        } else if let Ok(text) = std::str::from_utf8(&bytes) {
            text.to_owned()
        } else {
            // Preserve ureq's lossy text decoding for an unlabeled, malformed response.
            ureq::Body::builder()
                .mime_type(body.mime_type().unwrap_or(""))
                .data(bytes.clone())
                .read_to_string()
                .map_err(|e| format!("Failed to decode response from {url}: {e}"))?
        };

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "http".to_string());

        // Check if content looks like HTML
        let (text, source_map) = if deformat::detect::is_html(&content) {
            metadata.insert("content-type".to_string(), "text/html".to_string());
            if self.read_declared_language {
                if let Some(language) = deformat::html::extract_metadata(&content).language {
                    metadata.insert("declared_language".to_owned(), language);
                }
            }
            // Routes between readability / html2text based on anno-cli features.
            let result = crate::cli::utils::extract_html(&content, Some(&source_url));
            metadata.insert("extractor".to_string(), result.extractor.to_string());
            if let Some(title) = &result.title {
                metadata.insert("title".to_string(), title.clone());
            }
            if result.fallback {
                metadata.insert("fallback".to_string(), "true".to_string());
            }
            let map = if content.as_bytes() == bytes {
                deformat::mapped::html_source_map(&content, &result.text, result.extractor)?
            } else {
                anno::SourceMap::unmapped(&result.text, Some(&bytes))
            };
            (result.text, map)
        } else {
            metadata.insert("content-type".to_string(), "text/plain".to_string());
            let map = if content.as_bytes() == bytes {
                anno::SourceMap::identity(&content)
            } else {
                anno::SourceMap::unmapped(&content, Some(&bytes))
            };
            (content, map)
        };
        Ok(ResolvedContent {
            source_map,
            text,
            metadata,
            source_url,
        })
    }
}

/// Composite resolver that tries multiple resolvers in order.
pub struct CompositeResolver {
    resolvers: Vec<Box<dyn UrlResolver>>,
}

impl std::fmt::Debug for CompositeResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeResolver")
            .field("resolver_count", &self.resolvers.len())
            .finish()
    }
}

impl CompositeResolver {
    /// Create a new composite resolver with default resolvers.
    #[must_use]
    pub fn new() -> Self {
        Self::with_declared_language(false)
    }

    pub(crate) fn with_declared_language(read_declared_language: bool) -> Self {
        let mut http = HttpResolver::new();
        http.read_declared_language = read_declared_language;
        let resolvers = vec![Box::new(http) as Box<dyn UrlResolver>];
        Self { resolvers }
    }
}

impl Default for CompositeResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl UrlResolver for CompositeResolver {
    fn can_resolve(&self, url: &str) -> bool {
        self.resolvers.iter().any(|r| r.can_resolve(url))
    }

    fn resolve(&self, url: &str) -> Result<ResolvedContent, String> {
        for resolver in &self.resolvers {
            if resolver.can_resolve(url) {
                return resolver.resolve(url);
            }
        }
        Err(format!("No resolver available for URL: {}", url))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicitly_configured_ca_file_must_provide_certificates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(http_agent(None).is_ok());
        assert!(http_agent(Some(&root.join("missing-ca-bundle.pem")))
            .unwrap_err()
            .contains("Cannot read SSL_CERT_FILE"));
        assert!(http_agent(Some(&root.join("Cargo.toml")))
            .unwrap_err()
            .contains("No certificates in SSL_CERT_FILE"));
    }

    #[test]
    fn http_resolver_can_resolve_http() {
        let resolver = HttpResolver::new();
        assert!(resolver.can_resolve("http://example.com"));
        assert!(resolver.can_resolve("https://example.com"));
        assert!(resolver.can_resolve("http://example.com/path?query=1"));
        assert!(resolver.can_resolve("https://subdomain.example.com/path"));
    }

    fn resolve_body(content_type: &'static str, bytes: &[u8]) -> ResolvedContent {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let bytes = bytes.to_vec();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            for line in BufReader::new(&stream).lines() {
                if line.unwrap().is_empty() {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
                bytes.len()
            )
            .unwrap();
            stream.write_all(&bytes).unwrap();
        });
        let resolved = HttpResolver::new().resolve(&url).unwrap();
        server.join().unwrap();
        resolved
    }

    #[test]
    fn http_plain_text_source_map_identifies_body_before_decoding() {
        let original = " \t日本\r\n";
        let plain = resolve_body("text/plain; charset=utf-8", original.as_bytes());
        assert_eq!(plain.text, original);
        assert_eq!(plain.source_map, anno::SourceMap::identity(original));

        let original_bytes = b"\x93\xfa\x96\x7b"; // 日本 in Shift_JIS
        let decoded = resolve_body("text/plain; charset=Shift_JIS", original_bytes);
        assert_eq!(decoded.text, "日本");
        assert_eq!(
            decoded.source_map,
            anno::SourceMap::unmapped(&decoded.text, Some(original_bytes))
        );
    }

    #[test]
    fn http_resolver_decodes_shift_jis_html() {
        for (content_type, meta) in [
            ("TeXt/HtMl", "Shift_JIS"),
            ("text/html; CHARSET=\"Shift_JIS\"", "utf-8"),
            ("application/xhtml+xml; charset=Shift_JIS", "utf-8"),
        ] {
            let mut bytes =
                format!("<html><head><meta charset='{meta}'></head><body><p>").into_bytes();
            bytes.extend_from_slice(b"\x93\xfa\x96\x7b</p></body></html>"); // 日本 in Shift_JIS
            let resolved = resolve_body(content_type, &bytes);
            assert_eq!(
                resolved.source_map,
                anno::SourceMap::unmapped(&resolved.text, Some(&bytes))
            );
            assert!(
                resolved.text.contains("日本"),
                "{content_type}: {}",
                resolved.text
            );
        }
    }

    #[test]
    fn http_resolver_case_sensitive() {
        let resolver = HttpResolver::new();
        assert!(!resolver.can_resolve("HTTP://example.com"));
        assert!(!resolver.can_resolve("HTTPS://example.com"));
    }

    #[test]
    fn http_resolver_cannot_resolve_other_schemes() {
        let resolver = HttpResolver::new();
        assert!(!resolver.can_resolve("ftp://example.com"));
        assert!(!resolver.can_resolve("file:///path/to/file"));
        assert!(!resolver.can_resolve("mailto:test@example.com"));
        assert!(!resolver.can_resolve("not_a_url"));
    }

    #[test]
    fn resolved_content_struct() {
        let content = ResolvedContent {
            text: "Hello world".to_string(),
            source_map: anno::SourceMap::identity("Hello world"),
            metadata: HashMap::new(),
            source_url: "https://example.com".to_string(),
        };
        assert_eq!(content.text, "Hello world");
        assert!(content.metadata.is_empty());
        assert_eq!(content.source_url, "https://example.com");
    }

    #[test]
    fn composite_resolver_creation() {
        let resolver = CompositeResolver::new();
        assert!(resolver.can_resolve("https://example.com"));
    }

    #[test]
    fn composite_resolver_default() {
        let resolver = CompositeResolver::default();
        assert!(resolver.can_resolve("http://example.com"));
    }

    #[test]
    fn composite_resolver_cannot_resolve_unknown() {
        let resolver = CompositeResolver::new();
        assert!(!resolver.can_resolve("custom://unknown"));
    }

    #[test]
    fn composite_resolver_no_matching_resolver() {
        let resolver = CompositeResolver { resolvers: vec![] };
        let result = resolver.resolve("any://url");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No resolver available"));
    }

    #[test]
    fn composite_resolver_debug() {
        let resolver = CompositeResolver::new();
        let debug = format!("{:?}", resolver);
        assert!(debug.contains("CompositeResolver"));
        assert!(debug.contains("resolver_count"));
    }

    #[test]
    fn http_resolver_debug() {
        let resolver = HttpResolver::new();
        let debug = format!("{:?}", resolver);
        assert!(debug.contains("HttpResolver"));
    }

    #[test]
    fn resolved_content_clone() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());
        let content = ResolvedContent {
            text: "test".to_string(),
            source_map: anno::SourceMap::identity("test"),
            metadata,
            source_url: "http://test.com".to_string(),
        };
        let cloned = content.clone();
        assert_eq!(content.text, cloned.text);
        assert_eq!(content.source_url, cloned.source_url);
        assert_eq!(content.metadata, cloned.metadata);
    }

    // =========================================================================
    // Readability extraction
    // =========================================================================

    #[cfg(feature = "extractor-readability")]
    #[test]
    fn readability_extracts_article_text() {
        // dom_smoothie needs a substantial article body to trigger extraction.
        // Each paragraph must be long enough to score above the readability threshold.
        let html = r#"<!DOCTYPE html>
        <html><head><title>Breaking News: Scientists Discover New Species</title></head>
        <body>
            <nav><a href="/">Home</a><a href="/about">About</a><a href="/contact">Contact</a></nav>
            <div id="content">
                <h1>Breaking News: Scientists Discover New Species</h1>
                <p>A team of researchers at the University of Cambridge has announced
                   the discovery of a previously unknown species of beetle in the
                   Amazon rainforest. The discovery was published in the journal
                   Nature on March 15, 2026. The finding represents one of the most
                   significant entomological discoveries in the region in recent years,
                   and has drawn attention from conservation organizations worldwide.</p>
                <p>Lead researcher Dr. Sarah Chen said the species, named Chrysina
                   amazonica, was found during an expedition in January near the city
                   of Manaus. The beetle has unique iridescent markings that distinguish
                   it from related species in the genus. Chen and her team spent three
                   weeks collecting specimens and documenting the habitat conditions
                   where the species was found along tributary streams of the Amazon River.</p>
                <p>The Amazon rainforest continues to yield new discoveries despite
                   decades of intensive exploration by international research teams.
                   Conservation groups including the World Wildlife Fund and the
                   International Union for Conservation of Nature have called for
                   increased protection of the region. Brazil's Environment Ministry
                   said it would review the protected area boundaries in light of
                   the new findings. Local indigenous communities have long known
                   about the beetle but it had never been formally described by science.</p>
                <p>The research was funded by a grant from the European Research Council
                   and the National Geographic Society. Additional specimens will be
                   housed at the Natural History Museum in London and the Smithsonian
                   Institution in Washington, D.C. Future expeditions are planned for
                   July 2026 to search for related species in neighboring regions of
                   Peru and Colombia.</p>
            </div>
            <footer><p>Copyright 2026 News Corp. All rights reserved.</p></footer>
        </body></html>"#;

        let result = deformat::html::extract_with_readability(html, "https://example.com/article");
        assert!(result.is_some(), "readability should extract article text");
        let (text, title, _excerpt) = result.unwrap();
        assert!(
            text.contains("Dr. Sarah Chen"),
            "should contain person name, got: {}",
            text
        );
        assert!(
            text.contains("University of Cambridge"),
            "should contain org, got: {}",
            text
        );
        assert!(title.is_some(), "should extract title");
    }

    #[cfg(feature = "extractor-readability")]
    #[test]
    fn readability_returns_none_for_minimal_html() {
        let html = "<html><body><p>Hi</p></body></html>";
        let result = deformat::html::extract_with_readability(html, "https://example.com");
        assert!(
            result.is_none(),
            "should return None for trivial HTML (<50 chars)"
        );
    }

    #[cfg(feature = "extractor-readability")]
    #[test]
    fn readability_returns_none_for_empty_html() {
        let result = deformat::html::extract_with_readability("", "https://example.com");
        assert!(result.is_none(), "should return None for empty HTML");
    }

    #[cfg(feature = "extractor-readability")]
    #[test]
    fn readability_returns_none_for_nav_only_page() {
        let html = r#"<html><body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
        </body></html>"#;
        let result = deformat::html::extract_with_readability(html, "https://example.com");
        assert!(result.is_none(), "should return None for nav-only page");
    }
}
