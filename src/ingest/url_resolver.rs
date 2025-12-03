//! URL resolution connectors for fetching content from URLs.
//!
//! Provides a trait-based system for resolving different URL types to text content.

use crate::Result;
use std::collections::HashMap;

/// Resolved content from a URL.
#[derive(Debug, Clone)]
pub struct ResolvedContent {
    /// The extracted text content
    pub text: String,
    /// Metadata about the source (title, content-type, etc.)
    pub metadata: HashMap<String, String>,
    /// The original URL
    pub source_url: String,
}

/// Trait for URL resolvers that can fetch and extract text from URLs.
pub trait UrlResolver: std::fmt::Debug {
    /// Check if this resolver can handle the given URL.
    fn can_resolve(&self, url: &str) -> bool;

    /// Resolve the URL to text content.
    fn resolve(&self, url: &str) -> Result<ResolvedContent>;
}

/// HTTP/HTTPS URL resolver.
///
/// Fetches content from HTTP/HTTPS URLs and extracts text from HTML if needed.
#[derive(Debug, Default)]
pub struct HttpResolver;

impl HttpResolver {
    /// Create a new HTTP resolver.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract text from HTML content (simple, no full HTML parser).
    ///
    /// Removes HTML tags and decodes common entities.
    fn extract_text_from_html(&self, html: &str) -> String {
        let mut text = String::with_capacity(html.len());
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;
        let mut chars = html.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '<' => {
                    in_tag = true;
                    // Check for script/style tags
                    let mut tag_buffer = String::new();
                    tag_buffer.push('<');
                    let mut tag_name = String::new();
                    let mut in_tag_name = true;

                    while let Some(&next_ch) = chars.peek() {
                        if next_ch == '>' {
                            chars.next();
                            tag_buffer.push('>');
                            let tag_lower = tag_name.to_lowercase();
                            if tag_lower == "script" || tag_lower.starts_with("script ") {
                                in_script = true;
                            } else if tag_lower == "/script" || tag_lower.starts_with("/script ") {
                                in_script = false;
                            } else if tag_lower == "style" || tag_lower.starts_with("style ") {
                                in_style = true;
                            } else if tag_lower == "/style" || tag_lower.starts_with("/style ") {
                                in_style = false;
                            }
                            in_tag = false;
                            break;
                        } else if next_ch.is_whitespace() {
                            in_tag_name = false;
                            tag_buffer.push(chars.next().unwrap());
                        } else if in_tag_name {
                            tag_name.push(chars.next().unwrap());
                        } else {
                            tag_buffer.push(chars.next().unwrap());
                        }
                    }
                    // Don't add script/style content
                    if !in_script && !in_style {
                        // Add space after block elements for readability
                        if matches!(
                            tag_name.to_lowercase().as_str(),
                            "p" | "div" | "br" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                        ) {
                            if !text.ends_with(' ') && !text.is_empty() {
                                text.push(' ');
                            }
                        }
                    }
                }
                '>' if in_tag => {
                    in_tag = false;
                }
                _ if in_tag || in_script || in_style => {
                    // Skip content inside tags, scripts, styles
                }
                '&' => {
                    // Decode common HTML entities
                    let mut entity = String::new();
                    entity.push('&');
                    let mut found_semicolon = false;
                    while let Some(&next_ch) = chars.peek() {
                        entity.push(chars.next().unwrap());
                        if next_ch == ';' {
                            found_semicolon = true;
                            break;
                        }
                        if next_ch.is_whitespace() || next_ch == '<' {
                            break;
                        }
                    }

                    if found_semicolon {
                        let decoded = match entity.as_str() {
                            "&amp;" => "&",
                            "&lt;" => "<",
                            "&gt;" => ">",
                            "&quot;" => "\"",
                            "&apos;" => "'",
                            "&nbsp;" => " ",
                            "&#39;" => "'",
                            "&#8217;" => "'",
                            "&#8220;" => "\"",
                            "&#8221;" => "\"",
                            _ => {
                                // Try numeric entity
                                if entity.starts_with("&#") && entity.len() > 2 {
                                    let num_str = &entity[2..entity.len() - 1];
                                    if let Ok(num) = num_str.parse::<u32>() {
                                        if let Some(ch) = char::from_u32(num) {
                                            text.push(ch);
                                            continue;
                                        }
                                    }
                                }
                                // Unknown entity, keep as-is
                                text.push_str(&entity);
                                continue;
                            }
                        };
                        text.push_str(decoded);
                    } else {
                        // Not a valid entity, keep as-is
                        text.push('&');
                        text.push_str(&entity[1..]);
                    }
                }
                ch if !in_tag && !in_script && !in_style => {
                    text.push(ch);
                }
                _ => {}
            }
        }

        // Clean up whitespace
        text.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl UrlResolver for HttpResolver {
    fn can_resolve(&self, url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }

    fn resolve(&self, url: &str) -> Result<ResolvedContent> {
        #[cfg(feature = "eval-advanced")]
        {
            // Reuse the download infrastructure from eval/loader
            // Note: download_attempt is private, so we'll implement our own
            let response = ureq::get(url)
                .timeout(std::time::Duration::from_secs(60))
                .call()
                .map_err(|e| {
                    let error_msg = format!("{}", e);
                    crate::Error::InvalidInput(format!(
                        "Network error fetching {}: {}. \
                         Check your internet connection and try again.",
                        url, error_msg
                    ))
                })?;

            if response.status() != 200 {
                return Err(crate::Error::InvalidInput(format!(
                    "HTTP {} fetching {}. \
                     Server returned error status. \
                     URL may be temporarily unavailable or changed.",
                    response.status(),
                    url
                )));
            }

            let content = response.into_string().map_err(|e| {
                crate::Error::InvalidInput(format!(
                    "Failed to read response from {}: {}. \
                     Response may be too large or corrupted.",
                    url, e
                ))
            })?;

            let mut metadata = HashMap::new();
            metadata.insert("content-type".to_string(), "text/html".to_string());
            metadata.insert("source".to_string(), "http".to_string());

            // Check if content looks like HTML
            let text = if content.trim_start().starts_with('<') {
                // HTML content - extract text
                metadata.insert("content-type".to_string(), "text/html".to_string());
                self.extract_text_from_html(&content)
            } else {
                // Plain text
                metadata.insert("content-type".to_string(), "text/plain".to_string());
                content
            };

            Ok(ResolvedContent {
                text,
                metadata,
                source_url: url.to_string(),
            })
        }

        #[cfg(not(feature = "eval-advanced"))]
        {
            Err(crate::Error::InvalidInput(
                "URL resolution requires 'eval-advanced' feature. \
                 Enable it with: cargo build --features eval-advanced"
                    .to_string(),
            ))
        }
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
        let mut resolvers: Vec<Box<dyn UrlResolver>> = Vec::new();
        resolvers.push(Box::new(HttpResolver::new()));
        Self { resolvers }
    }

    /// Add a resolver to the chain.
    pub fn add_resolver(&mut self, resolver: Box<dyn UrlResolver>) {
        self.resolvers.push(resolver);
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

    fn resolve(&self, url: &str) -> Result<ResolvedContent> {
        for resolver in &self.resolvers {
            if resolver.can_resolve(url) {
                return resolver.resolve(url);
            }
        }
        Err(crate::Error::InvalidInput(format!(
            "No resolver available for URL: {}",
            url
        )))
    }
}
