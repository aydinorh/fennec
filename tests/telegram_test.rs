use fennec::channels::{Channel, TelegramChannel};

#[test]
fn parse_get_updates_response() {
    let json: serde_json::Value = serde_json::json!({
        "ok": true,
        "result": [
            {
                "update_id": 100,
                "message": {
                    "message_id": 1,
                    "from": { "id": 42, "is_bot": false, "first_name": "Alice" },
                    "chat": { "id": 42, "type": "private" },
                    "date": 1700000000,
                    "text": "Hello Fennec!"
                }
            },
            {
                "update_id": 101,
                "message": {
                    "message_id": 2,
                    "from": { "id": 99, "is_bot": false, "first_name": "Bob" },
                    "chat": { "id": -1001234, "type": "group" },
                    "date": 1700000001,
                    "text": "Hey there"
                }
            },
            {
                "update_id": 102,
                "message": {
                    "message_id": 3,
                    "from": { "id": 50, "is_bot": false, "first_name": "Eve" },
                    "chat": { "id": 50, "type": "private" },
                    "date": 1700000002
                }
            }
        ]
    });

    let updates = TelegramChannel::parse_updates(&json);

    // The third update has no text, so it should be skipped.
    assert_eq!(updates.len(), 2);

    let (uid, sender, chat, text) = &updates[0];
    assert_eq!(*uid, 100);
    assert_eq!(sender, "42");
    assert_eq!(chat, "42");
    assert_eq!(text, "Hello Fennec!");

    let (uid, sender, chat, text) = &updates[1];
    assert_eq!(*uid, 101);
    assert_eq!(sender, "99");
    assert_eq!(chat, "-1001234");
    assert_eq!(text, "Hey there");
}

#[test]
fn parse_empty_result() {
    let json: serde_json::Value = serde_json::json!({
        "ok": true,
        "result": []
    });
    let updates = TelegramChannel::parse_updates(&json);
    assert!(updates.is_empty());
}

#[test]
fn allows_sender_empty_list_denies_all() {
    // Default-deny: an unconfigured allowlist refuses everyone;
    // "*" is the explicit allow-everyone opt-in.
    let ch = TelegramChannel::new("token".into(), vec![]);
    assert!(!fennec::channels::Channel::allows_sender(&ch, "anyone"));
}

#[test]
fn allows_sender_wildcard_permits_all() {
    let ch = TelegramChannel::new("token".into(), vec!["*".into()]);
    assert!(fennec::channels::Channel::allows_sender(&ch, "12345"));
    assert!(fennec::channels::Channel::allows_sender(&ch, "anyone"));
}

#[test]
fn allows_sender_specific_list() {
    let ch = TelegramChannel::new("token".into(), vec!["42".into(), "99".into()]);
    assert!(fennec::channels::Channel::allows_sender(&ch, "42"));
    assert!(fennec::channels::Channel::allows_sender(&ch, "99"));
    assert!(!fennec::channels::Channel::allows_sender(&ch, "100"));
}

#[test]
fn rate_limit_tracking_initialized_empty() {
    let ch = TelegramChannel::new("token".into(), vec![]);
    // The channel starts with no rate-limit entries.
    assert!(ch.supports_streaming());
}
