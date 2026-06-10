use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::bus::InboundMessage;

use super::traits::{Channel, SendMessage};

/// Email channel with IMAP polling for incoming mail and SMTP (via lettre) for
/// outgoing messages.
pub struct EmailChannel {
    imap_host: String,
    imap_port: u16,
    imap_user: String,
    imap_password: String,
    smtp_host: String,
    smtp_port: u16,
    smtp_user: String,
    smtp_password: String,
    from_address: String,
    allowed_senders: Vec<String>,
    poll_interval_secs: u64,
}

impl EmailChannel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        imap_host: String,
        imap_port: u16,
        imap_user: String,
        imap_password: String,
        smtp_host: String,
        smtp_port: u16,
        smtp_user: String,
        smtp_password: String,
        from_address: String,
        allowed_senders: Vec<String>,
        poll_interval_secs: u64,
    ) -> Self {
        Self {
            imap_host,
            imap_port,
            imap_user,
            imap_password,
            smtp_host,
            smtp_port,
            smtp_user,
            smtp_password,
            from_address,
            allowed_senders,
            poll_interval_secs: if poll_interval_secs == 0 {
                30
            } else {
                poll_interval_secs
            },
        }
    }

    /// Extract the plain-text sender email from a raw "From" header value.
    /// Handles formats like:
    ///   "Alice <alice@example.com>"
    ///   "alice@example.com"
    pub fn extract_sender_email(from: &str) -> String {
        if let Some(start) = from.find('<') {
            if let Some(end) = from.find('>') {
                return from[start + 1..end].trim().to_lowercase();
            }
        }
        from.trim().to_lowercase()
    }

    /// Extract the plain-text body from a raw email body string.
    /// This is a best-effort parser: it takes the first text/plain part
    /// or falls back to the raw body with tags stripped.
    pub fn extract_text_body(raw: &str) -> String {
        // Simple approach: just use the raw body, trimming common MIME artifacts.
        // A production implementation would use a proper MIME parser.
        let body = raw.trim();
        if body.is_empty() {
            return String::new();
        }
        body.to_string()
    }

    /// Connect to IMAP and fetch unseen messages via blocking I/O in a
    /// spawned thread. Returns (sender, subject, body) tuples.
    async fn fetch_unseen(&self) -> Result<Vec<(String, String, String)>> {
        let host = self.imap_host.clone();
        let port = self.imap_port;
        let user = self.imap_user.clone();
        let password = self.imap_password.clone();

        tokio::task::spawn_blocking(move || {
            let tls = native_tls::TlsConnector::new()
                .context("Failed to create TLS connector")?;
            let addr = format!("{}:{}", host, port);
            let tcp = std::net::TcpStream::connect(&addr)
                .context("IMAP TCP connect failed")?;
            tcp.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
            let tls_stream = tls.connect(&host, tcp)
                .context("IMAP TLS connect failed")?;

            // Simple IMAP implementation using raw TLS stream.
            // We use a single stream for both read and write by wrapping
            // in a BufReader that owns the stream.
            use std::io::{BufRead, BufReader, Write};
            let mut stream = BufReader::new(tls_stream);
            let tag = std::cell::Cell::new(1u32);

            let send_cmd = |stream: &mut BufReader<native_tls::TlsStream<std::net::TcpStream>>, cmd: &str| -> Result<Vec<String>> {
                let t = tag.get();
                let tagged = format!("A{} {}\r\n", t, cmd);
                tag.set(t + 1);
                stream.get_mut().write_all(tagged.as_bytes())?;
                stream.get_mut().flush()?;

                let mut lines = Vec::new();
                loop {
                    let mut line = String::new();
                    stream.read_line(&mut line)?;
                    let done = line.starts_with(&format!("A{} ", t));
                    lines.push(line);
                    if done { break; }
                }
                Ok(lines)
            };

            // Read greeting
            let mut greeting = String::new();
            stream.read_line(&mut greeting)?;

            // LOGIN
            let login_resp = send_cmd(&mut stream, &format!("LOGIN \"{}\" \"{}\"", user, password))?;
            if let Some(last) = login_resp.last() {
                if !last.contains("OK") {
                    anyhow::bail!("IMAP login failed: {}", last.trim());
                }
            }

            // SELECT INBOX
            send_cmd(&mut stream, "SELECT INBOX")?;

            // SEARCH UNSEEN
            let search_resp = send_cmd(&mut stream, "SEARCH UNSEEN")?;
            let mut msg_nums: Vec<String> = Vec::new();
            for line in &search_resp {
                if line.starts_with("* SEARCH") {
                    msg_nums = line.trim_start_matches("* SEARCH")
                        .trim()
                        .split_whitespace()
                        .map(String::from)
                        .filter(|s| !s.is_empty())
                        .collect();
                }
            }

            if msg_nums.is_empty() {
                let _ = send_cmd(&mut stream, "LOGOUT");
                return Ok(Vec::new());
            }

            let mut results = Vec::new();

            for num in &msg_nums {
                let fetch_resp = send_cmd(&mut stream, &format!("FETCH {} BODY[HEADER.FIELDS (FROM SUBJECT)]", num))?;
                let header_text: String = fetch_resp.iter()
                    .filter(|l| !l.starts_with("*") && !l.starts_with(&format!("A{}", tag.get() - 1)))
                    .cloned()
                    .collect();

                let mut sender = String::new();
                let mut subject = String::new();
                for line in header_text.lines() {
                    let lower = line.to_lowercase();
                    if lower.starts_with("from:") {
                        sender = Self::extract_sender_email(&line[5..]);
                    } else if lower.starts_with("subject:") {
                        subject = line[8..].trim().to_string();
                    }
                }

                // Fetch body
                let body_resp = send_cmd(&mut stream, &format!("FETCH {} BODY[TEXT]", num))?;
                let body: String = body_resp.iter()
                    .filter(|l| !l.starts_with("*") && !l.starts_with(&format!("A{}", tag.get() - 1)) && !l.starts_with(")"))
                    .cloned()
                    .collect();
                let body = Self::extract_text_body(&body);

                if !sender.is_empty() && !body.is_empty() {
                    results.push((sender, subject, body));
                }

                // Mark as seen
                let _ = send_cmd(&mut stream, &format!("STORE {} +FLAGS (\\Seen)", num));
            }

            let _ = send_cmd(&mut stream, "LOGOUT");
            Ok(results)
        }).await?
    }

    /// Send an email via SMTP using lettre.
    fn build_smtp_transport(
        &self,
    ) -> Result<lettre::AsyncSmtpTransport<lettre::Tokio1Executor>> {
        use lettre::transport::smtp::authentication::Credentials;
        use lettre::AsyncSmtpTransport;

        let creds = Credentials::new(self.smtp_user.clone(), self.smtp_password.clone());

        let transport = if self.smtp_port == 465 {
            AsyncSmtpTransport::<lettre::Tokio1Executor>::relay(&self.smtp_host)
                .context("SMTP relay setup failed")?
                .credentials(creds)
                .port(self.smtp_port)
                .build()
        } else {
            AsyncSmtpTransport::<lettre::Tokio1Executor>::starttls_relay(&self.smtp_host)
                .context("SMTP STARTTLS relay setup failed")?
                .credentials(creds)
                .port(self.smtp_port)
                .build()
        };

        Ok(transport)
    }
}

#[async_trait]
impl Channel for EmailChannel {
    fn name(&self) -> &str {
        "email"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        use lettre::message::header::ContentType;
        use lettre::Message;
        use lettre::AsyncTransport;

        let email = Message::builder()
            .from(
                self.from_address
                    .parse()
                    .context("Invalid from_address for SMTP")?,
            )
            .to(message
                .recipient
                .parse()
                .context("Invalid recipient email address")?)
            .subject("Re: Fennec")
            .header(ContentType::TEXT_PLAIN)
            .body(message.content.clone())
            .context("Failed to build email message")?;

        let transport = self.build_smtp_transport()?;
        transport
            .send(email)
            .await
            .context("SMTP send failed")?;

        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<InboundMessage>) -> Result<()> {
        let interval = tokio::time::Duration::from_secs(self.poll_interval_secs);

        loop {
            match self.fetch_unseen().await {
                Ok(messages) => {
                    for (sender, subject, body) in messages {
                        if !self.allows_sender(&sender) {
                            tracing::debug!(
                                "Email: ignoring message from disallowed sender {}",
                                sender
                            );
                            continue;
                        }

                        let content = if subject.is_empty() {
                            body
                        } else {
                            format!("[{}] {}", subject, body)
                        };

                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();

                        let msg = InboundMessage {
                            id: uuid::Uuid::new_v4().to_string(),
                            sender: sender.clone(),
                            content,
                            channel: "email".to_string(),
                            chat_id: sender,
                            timestamp: now,
                            reply_to: None,
                            metadata: HashMap::new(),
                        };

                        if tx.send(msg).await.is_err() {
                            tracing::info!("Email: inbound channel closed, stopping listener");
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Email: IMAP poll error: {}", e);
                }
            }

            tokio::time::sleep(interval).await;
        }
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        // Default-DENY: an empty allowlist refuses everyone ("*" opts
        // into allow-everyone). See the telegram channel for rationale.
        if self.allowed_senders.is_empty() {
            tracing::warn!(
                "Email: refusing message from '{sender_id}' — no allowed_senders configured. \
                 Add the address to [channels.email].allowed_senders (or \"*\" to allow everyone)."
            );
            return false;
        }
        if self.allowed_senders.iter().any(|u| u == "*") {
            return true;
        }
        let sender_lower = sender_id.to_lowercase();
        self.allowed_senders
            .iter()
            .any(|u| u.to_lowercase() == sender_lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_sender_email_with_angle_brackets() {
        assert_eq!(
            EmailChannel::extract_sender_email("Alice Smith <alice@example.com>"),
            "alice@example.com"
        );
    }

    #[test]
    fn test_extract_sender_email_plain() {
        assert_eq!(
            EmailChannel::extract_sender_email("alice@example.com"),
            "alice@example.com"
        );
    }

    #[test]
    fn test_extract_sender_email_uppercase() {
        assert_eq!(
            EmailChannel::extract_sender_email("BOB@Example.COM"),
            "bob@example.com"
        );
    }

    #[test]
    fn test_extract_text_body() {
        assert_eq!(
            EmailChannel::extract_text_body("  Hello world  "),
            "Hello world"
        );
        assert_eq!(EmailChannel::extract_text_body(""), "");
    }

    #[test]
    fn test_allows_sender_empty_denies() {
        // Default-deny: an unconfigured allowlist must refuse everyone;
        // "*" is the explicit allow-everyone opt-in.
        let ch = EmailChannel::new(
            "imap.example.com".to_string(),
            993,
            "user".to_string(),
            "pass".to_string(),
            "smtp.example.com".to_string(),
            587,
            "user".to_string(),
            "pass".to_string(),
            "bot@example.com".to_string(),
            vec![],
            30,
        );
        assert!(!ch.allows_sender("anyone@example.com"));
    }

    #[test]
    fn test_allows_sender_wildcard() {
        let ch = EmailChannel::new(
            "imap.example.com".to_string(),
            993,
            "user".to_string(),
            "pass".to_string(),
            "smtp.example.com".to_string(),
            587,
            "user".to_string(),
            "pass".to_string(),
            "bot@example.com".to_string(),
            vec!["*".to_string()],
            30,
        );
        assert!(ch.allows_sender("anyone@example.com"));
    }

    #[test]
    fn test_allows_sender_specific() {
        let ch = EmailChannel::new(
            "imap.example.com".to_string(),
            993,
            "user".to_string(),
            "pass".to_string(),
            "smtp.example.com".to_string(),
            587,
            "user".to_string(),
            "pass".to_string(),
            "bot@example.com".to_string(),
            vec!["alice@example.com".to_string()],
            30,
        );
        assert!(ch.allows_sender("alice@example.com"));
        assert!(ch.allows_sender("Alice@Example.COM")); // case insensitive
        assert!(!ch.allows_sender("bob@example.com"));
    }
}
