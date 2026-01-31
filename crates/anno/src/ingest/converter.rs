//! Document format conversion using multiple backends.
//!
//! Converts various document formats to plain text for entity extraction.
//! Supports multiple conversion tools with automatic fallback:
//!
//! - **pandoc**: Universal document converter (HTML, Markdown, DOCX, etc.)
//! - **pdftotext**: PDF text extraction (poppler-utils)
//! - **html2text**: Clean HTML to text conversion
//! - **lynx**: Terminal browser with -dump mode
//! - **antiword**: Old .doc file conversion
//! - **unrtf**: RTF to text
//!
//! Falls back gracefully when tools are not available.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Supported document formats for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// Plain text (no conversion needed)
    PlainText,
    /// HTML documents
    Html,
    /// Markdown
    Markdown,
    /// Microsoft Word (docx)
    Docx,
    /// PDF documents
    Pdf,
    /// Rich Text Format
    Rtf,
    /// LaTeX
    Latex,
    /// EPUB ebooks
    Epub,
    /// reStructuredText
    Rst,
    /// Org-mode
    Org,
    /// MediaWiki format
    MediaWiki,
    /// Unknown format
    Unknown,
}

impl DocumentFormat {
    /// Detect format from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "txt" | "text" => Self::PlainText,
            "html" | "htm" => Self::Html,
            "md" | "markdown" => Self::Markdown,
            "docx" => Self::Docx,
            "pdf" => Self::Pdf,
            "rtf" => Self::Rtf,
            "tex" | "latex" => Self::Latex,
            "epub" => Self::Epub,
            "rst" => Self::Rst,
            "org" => Self::Org,
            "wiki" | "mediawiki" => Self::MediaWiki,
            _ => Self::Unknown,
        }
    }

    /// Detect format from file path.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Get pandoc input format name.
    #[must_use]
    pub fn pandoc_format(&self) -> Option<&'static str> {
        match self {
            Self::Html => Some("html"),
            Self::Markdown => Some("markdown"),
            Self::Docx => Some("docx"),
            Self::Pdf => None, // pandoc can't read PDFs directly
            Self::Rtf => Some("rtf"),
            Self::Latex => Some("latex"),
            Self::Epub => Some("epub"),
            Self::Rst => Some("rst"),
            Self::Org => Some("org"),
            Self::MediaWiki => Some("mediawiki"),
            Self::PlainText | Self::Unknown => None,
        }
    }
}

/// Available conversion backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConverterBackend {
    /// pandoc - universal document converter
    Pandoc,
    /// pdftotext - PDF extraction (poppler)
    PdfToText,
    /// html2text - HTML to text
    Html2Text,
    /// lynx - terminal browser with -dump
    Lynx,
    /// antiword - old .doc files
    Antiword,
    /// unrtf - RTF to text
    Unrtf,
}

impl ConverterBackend {
    /// Get the command name for this backend.
    #[must_use]
    pub fn command(&self) -> &'static str {
        match self {
            Self::Pandoc => "pandoc",
            Self::PdfToText => "pdftotext",
            Self::Html2Text => "html2text",
            Self::Lynx => "lynx",
            Self::Antiword => "antiword",
            Self::Unrtf => "unrtf",
        }
    }

    /// Get supported input formats.
    #[must_use]
    pub fn supported_formats(&self) -> &[DocumentFormat] {
        match self {
            Self::Pandoc => &[
                DocumentFormat::Html,
                DocumentFormat::Markdown,
                DocumentFormat::Docx,
                DocumentFormat::Rtf,
                DocumentFormat::Latex,
                DocumentFormat::Epub,
                DocumentFormat::Rst,
                DocumentFormat::Org,
                DocumentFormat::MediaWiki,
            ],
            Self::PdfToText => &[DocumentFormat::Pdf],
            Self::Html2Text => &[DocumentFormat::Html],
            Self::Lynx => &[DocumentFormat::Html],
            Self::Antiword => &[], // Old .doc files (not in DocumentFormat enum)
            Self::Unrtf => &[DocumentFormat::Rtf],
        }
    }
}

/// Document converter using multiple backends.
///
/// Opportunistically uses available tools for converting various
/// document formats to plain text. Falls back gracefully when
/// tools are not installed.
#[derive(Debug, Clone)]
pub struct DocumentConverter {
    /// Available backends (checked at construction)
    available_backends: Vec<ConverterBackend>,
    /// Custom paths for backends (backend -> path)
    custom_paths: std::collections::HashMap<ConverterBackend, String>,
}

impl Default for DocumentConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentConverter {
    /// Create a new converter, detecting available backends.
    #[must_use]
    pub fn new() -> Self {
        let all_backends = [
            ConverterBackend::Pandoc,
            ConverterBackend::PdfToText,
            ConverterBackend::Html2Text,
            ConverterBackend::Lynx,
            ConverterBackend::Antiword,
            ConverterBackend::Unrtf,
        ];

        let available_backends: Vec<_> = all_backends
            .into_iter()
            .filter(|b| Self::check_available(b.command()))
            .collect();

        Self {
            available_backends,
            custom_paths: std::collections::HashMap::new(),
        }
    }

    /// Create a converter with a custom path for a backend.
    #[must_use]
    pub fn with_custom_path(mut self, backend: ConverterBackend, path: impl Into<String>) -> Self {
        let path = path.into();
        if Self::check_available(&path) {
            self.custom_paths.insert(backend, path);
            if !self.available_backends.contains(&backend) {
                self.available_backends.push(backend);
            }
        }
        self
    }

    /// Check if a command is available.
    fn check_available(cmd: &str) -> bool {
        Command::new(cmd)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Check if any converter is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        !self.available_backends.is_empty()
    }

    /// Get available backends.
    #[must_use]
    pub fn available_backends(&self) -> &[ConverterBackend] {
        &self.available_backends
    }

    /// Check if a specific backend is available.
    #[must_use]
    pub fn has_backend(&self, backend: ConverterBackend) -> bool {
        self.available_backends.contains(&backend)
    }

    /// Get the best backend for a format.
    #[must_use]
    pub fn best_backend_for(&self, format: DocumentFormat) -> Option<ConverterBackend> {
        // Priority order for each format
        let preferences: &[ConverterBackend] = match format {
            DocumentFormat::Pdf => &[ConverterBackend::PdfToText, ConverterBackend::Pandoc],
            DocumentFormat::Html => &[
                ConverterBackend::Html2Text,
                ConverterBackend::Pandoc,
                ConverterBackend::Lynx,
            ],
            DocumentFormat::Rtf => &[ConverterBackend::Unrtf, ConverterBackend::Pandoc],
            _ => &[ConverterBackend::Pandoc],
        };

        preferences
            .iter()
            .find(|b| self.available_backends.contains(b))
            .copied()
    }

    /// Get command path for a backend.
    fn get_command(&self, backend: ConverterBackend) -> &str {
        self.custom_paths
            .get(&backend)
            .map(|s| s.as_str())
            .unwrap_or_else(|| backend.command())
    }

    /// Convert document content to plain text.
    ///
    /// Returns `Ok(text)` if conversion succeeds, or the original content
    /// if no suitable converter is available.
    pub fn convert(&self, content: &[u8], format: DocumentFormat) -> Result<String, String> {
        // Plain text and unknown formats: just decode as UTF-8
        if matches!(format, DocumentFormat::PlainText | DocumentFormat::Unknown) {
            return String::from_utf8(content.to_vec())
                .map_err(|e| format!("Invalid UTF-8: {}", e));
        }

        // Find best backend for this format
        let Some(backend) = self.best_backend_for(format) else {
            return String::from_utf8(content.to_vec())
                .map_err(|_| format!("No converter available for {:?}", format));
        };

        // Run the appropriate converter
        match backend {
            ConverterBackend::Pandoc => {
                let Some(input_format) = format.pandoc_format() else {
                    return Err(format!("Format {:?} not supported by pandoc", format));
                };
                self.run_pandoc(content, input_format)
            }
            ConverterBackend::PdfToText => self.run_pdftotext(content),
            ConverterBackend::Html2Text => self.run_html2text(content),
            ConverterBackend::Lynx => self.run_lynx(content),
            ConverterBackend::Antiword => self.run_antiword(content),
            ConverterBackend::Unrtf => self.run_unrtf(content),
        }
    }

    /// Convert a file to plain text.
    pub fn convert_file(&self, path: &Path) -> Result<String, String> {
        let format = DocumentFormat::from_path(path);

        // Some tools work better with file paths than stdin
        if format == DocumentFormat::Pdf && self.has_backend(ConverterBackend::PdfToText) {
            return self.run_pdftotext_file(path);
        }

        let content = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
        self.convert(&content, format)
    }

    /// Run pandoc to convert content.
    fn run_pandoc(&self, content: &[u8], input_format: &str) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::Pandoc);
        let mut child = Command::new(cmd)
            .args(["-f", input_format, "-t", "plain", "--wrap=none"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn pandoc: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(content)
                .map_err(|e| format!("Failed to write to pandoc stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for pandoc: {}", e))?;

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("Pandoc output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("Pandoc failed: {}", stderr))
        }
    }

    /// Run pdftotext on content (via temp file).
    fn run_pdftotext(&self, content: &[u8]) -> Result<String, String> {
        // pdftotext requires a file, so we use a temp file
        let temp_dir = std::env::temp_dir();
        let temp_pdf = temp_dir.join(format!("anno_convert_{}.pdf", std::process::id()));

        std::fs::write(&temp_pdf, content)
            .map_err(|e| format!("Failed to write temp PDF: {}", e))?;

        let result = self.run_pdftotext_file(&temp_pdf);
        let _ = std::fs::remove_file(&temp_pdf);
        result
    }

    /// Run pdftotext on a file.
    fn run_pdftotext_file(&self, path: &Path) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::PdfToText);
        let output = Command::new(cmd)
            .args(["-layout", "-enc", "UTF-8"])
            .arg(path)
            .arg("-") // Output to stdout
            .output()
            .map_err(|e| format!("Failed to run pdftotext: {}", e))?;

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("pdftotext output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("pdftotext failed: {}", stderr))
        }
    }

    /// Run html2text on content.
    fn run_html2text(&self, content: &[u8]) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::Html2Text);
        let mut child = Command::new(cmd)
            .args(["-utf8", "-width", "1000"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn html2text: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(content)
                .map_err(|e| format!("Failed to write to html2text stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for html2text: {}", e))?;

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("html2text output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("html2text failed: {}", stderr))
        }
    }

    /// Run lynx -dump on content.
    fn run_lynx(&self, content: &[u8]) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::Lynx);

        // lynx reads from file, so we use a temp file
        let temp_dir = std::env::temp_dir();
        let temp_html = temp_dir.join(format!("anno_convert_{}.html", std::process::id()));

        std::fs::write(&temp_html, content)
            .map_err(|e| format!("Failed to write temp HTML: {}", e))?;

        let output = Command::new(cmd)
            .args(["-dump", "-nolist", "-width", "1000"])
            .arg(&temp_html)
            .output()
            .map_err(|e| format!("Failed to run lynx: {}", e))?;

        let _ = std::fs::remove_file(&temp_html);

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("lynx output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("lynx failed: {}", stderr))
        }
    }

    /// Run antiword on content.
    fn run_antiword(&self, content: &[u8]) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::Antiword);

        // antiword reads from file
        let temp_dir = std::env::temp_dir();
        let temp_doc = temp_dir.join(format!("anno_convert_{}.doc", std::process::id()));

        std::fs::write(&temp_doc, content)
            .map_err(|e| format!("Failed to write temp DOC: {}", e))?;

        let output = Command::new(cmd)
            .arg(&temp_doc)
            .output()
            .map_err(|e| format!("Failed to run antiword: {}", e))?;

        let _ = std::fs::remove_file(&temp_doc);

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("antiword output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("antiword failed: {}", stderr))
        }
    }

    /// Run unrtf on content.
    fn run_unrtf(&self, content: &[u8]) -> Result<String, String> {
        let cmd = self.get_command(ConverterBackend::Unrtf);
        let mut child = Command::new(cmd)
            .args(["--text", "--nopict"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn unrtf: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(content)
                .map_err(|e| format!("Failed to write to unrtf stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for unrtf: {}", e))?;

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| format!("unrtf output is not valid UTF-8: {}", e))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("unrtf failed: {}", stderr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            DocumentFormat::from_extension("txt"),
            DocumentFormat::PlainText
        );
        assert_eq!(DocumentFormat::from_extension("html"), DocumentFormat::Html);
        assert_eq!(
            DocumentFormat::from_extension("md"),
            DocumentFormat::Markdown
        );
        assert_eq!(DocumentFormat::from_extension("docx"), DocumentFormat::Docx);
        assert_eq!(
            DocumentFormat::from_extension("xyz"),
            DocumentFormat::Unknown
        );
    }

    #[test]
    fn test_pandoc_format_names() {
        assert_eq!(DocumentFormat::Html.pandoc_format(), Some("html"));
        assert_eq!(DocumentFormat::Markdown.pandoc_format(), Some("markdown"));
        assert_eq!(DocumentFormat::PlainText.pandoc_format(), None);
        assert_eq!(DocumentFormat::Pdf.pandoc_format(), None); // pandoc can't read PDF
    }

    #[test]
    fn test_plain_text_passthrough() {
        let converter = DocumentConverter::new();
        let text = b"Hello, world!";
        let result = converter.convert(text, DocumentFormat::PlainText);
        assert_eq!(result.unwrap(), "Hello, world!");
    }

    #[test]
    fn test_converter_creation() {
        let converter = DocumentConverter::new();
        // Just verify it doesn't panic - pandoc may or may not be available
        let _ = converter.is_available();
    }

    #[test]
    fn test_format_detection_all_types() {
        // Test all supported extensions
        let test_cases = [
            ("txt", DocumentFormat::PlainText),
            ("text", DocumentFormat::PlainText),
            ("html", DocumentFormat::Html),
            ("htm", DocumentFormat::Html),
            ("md", DocumentFormat::Markdown),
            ("markdown", DocumentFormat::Markdown),
            ("docx", DocumentFormat::Docx),
            ("pdf", DocumentFormat::Pdf),
            ("rtf", DocumentFormat::Rtf),
            ("tex", DocumentFormat::Latex),
            ("latex", DocumentFormat::Latex),
            ("epub", DocumentFormat::Epub),
            ("rst", DocumentFormat::Rst),
            ("org", DocumentFormat::Org),
            ("wiki", DocumentFormat::MediaWiki),
            ("mediawiki", DocumentFormat::MediaWiki),
            ("xyz", DocumentFormat::Unknown),
            ("", DocumentFormat::Unknown),
        ];

        for (ext, expected) in test_cases {
            assert_eq!(
                DocumentFormat::from_extension(ext),
                expected,
                "Extension '{}' should map to {:?}",
                ext,
                expected
            );
        }
    }

    #[test]
    fn test_format_detection_case_insensitive() {
        assert_eq!(DocumentFormat::from_extension("HTML"), DocumentFormat::Html);
        assert_eq!(DocumentFormat::from_extension("PDF"), DocumentFormat::Pdf);
        assert_eq!(
            DocumentFormat::from_extension("Markdown"),
            DocumentFormat::Markdown
        );
    }

    #[test]
    fn test_format_from_path() {
        use std::path::PathBuf;

        let path = PathBuf::from("/some/path/document.html");
        assert_eq!(DocumentFormat::from_path(&path), DocumentFormat::Html);

        let path = PathBuf::from("file.pdf");
        assert_eq!(DocumentFormat::from_path(&path), DocumentFormat::Pdf);

        let path = PathBuf::from("no_extension");
        assert_eq!(DocumentFormat::from_path(&path), DocumentFormat::Unknown);
    }

    #[test]
    fn test_converter_backend_commands() {
        assert_eq!(ConverterBackend::Pandoc.command(), "pandoc");
        assert_eq!(ConverterBackend::PdfToText.command(), "pdftotext");
        assert_eq!(ConverterBackend::Html2Text.command(), "html2text");
        assert_eq!(ConverterBackend::Lynx.command(), "lynx");
        assert_eq!(ConverterBackend::Antiword.command(), "antiword");
        assert_eq!(ConverterBackend::Unrtf.command(), "unrtf");
    }

    #[test]
    fn test_converter_backend_supported_formats() {
        let pandoc_formats = ConverterBackend::Pandoc.supported_formats();
        assert!(pandoc_formats.contains(&DocumentFormat::Html));
        assert!(pandoc_formats.contains(&DocumentFormat::Markdown));
        assert!(!pandoc_formats.contains(&DocumentFormat::Pdf)); // pandoc can't read PDF

        let pdf_formats = ConverterBackend::PdfToText.supported_formats();
        assert!(pdf_formats.contains(&DocumentFormat::Pdf));
        assert_eq!(pdf_formats.len(), 1);
    }

    #[test]
    fn test_unknown_format_passthrough() {
        let converter = DocumentConverter::new();
        let text = b"Some content";
        let result = converter.convert(text, DocumentFormat::Unknown);
        assert_eq!(result.unwrap(), "Some content");
    }

    #[test]
    fn test_invalid_utf8() {
        let converter = DocumentConverter::new();
        let invalid_utf8 = vec![0xFF, 0xFE, 0x00, 0x01];
        let result = converter.convert(&invalid_utf8, DocumentFormat::PlainText);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("UTF-8"));
    }

    #[test]
    fn test_document_format_debug() {
        let format = DocumentFormat::Html;
        let debug = format!("{:?}", format);
        assert!(debug.contains("Html"));
    }

    #[test]
    fn test_document_format_clone_eq() {
        let format1 = DocumentFormat::Pdf;
        let format2 = format1;
        assert_eq!(format1, format2);
    }

    #[test]
    fn test_converter_backend_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ConverterBackend::Pandoc);
        set.insert(ConverterBackend::PdfToText);
        set.insert(ConverterBackend::Pandoc); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_converter_default() {
        let converter = DocumentConverter::default();
        // Should be same as new()
        let _ = converter.available_backends();
    }

    #[test]
    fn test_has_backend() {
        let converter = DocumentConverter::new();
        // Results depend on installed tools, but method should not panic
        let _ = converter.has_backend(ConverterBackend::Pandoc);
        let _ = converter.has_backend(ConverterBackend::PdfToText);
    }

    #[test]
    fn test_best_backend_for_format() {
        let converter = DocumentConverter::new();
        // Results depend on installed tools, but method should not panic
        let _ = converter.best_backend_for(DocumentFormat::Pdf);
        let _ = converter.best_backend_for(DocumentFormat::Html);
        let _ = converter.best_backend_for(DocumentFormat::Markdown);
    }
}
