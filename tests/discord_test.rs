use fennec::channels::{Channel, DiscordChannel};
use fennec::channels::discord::DISCORD_INTENTS;

#[test]
fn parse_message_create_payload() {
    let payload = serde_json::json!({
        "id": "1234567890",
        "type": 0,
        "content": "Hello from Discord!",
        "channel_id": "9876543210",
        "author": {
            "id": "111222333",
            "username": "alice",
            "bot": false
        },
        "timestamp": "2025-01-01T00:00:00.000000+00:00"
    });

    let result = DiscordChannel::parse_message_create(&payload);
    assert!(result.is_some());
    let (author_id, channel_id, content, msg_id, is_bot) = result.unwrap();
    assert_eq!(author_id, "111222333");
    assert_eq!(channel_id, "9876543210");
    assert_eq!(content, "Hello from Discord!");
    assert_eq!(msg_id, "1234567890");
    assert!(!is_bot);
}

#[test]
fn parse_message_create_bot_author() {
    let payload = serde_json::json!({
        "id": "999",
        "content": "Bot message",
        "channel_id": "555",
        "author": {
            "id": "666",
            "username": "some-bot",
            "bot": true
        }
    });

    let result = DiscordChannel::parse_message_create(&payload);
    assert!(result.is_some());
    let (_author_id, _channel_id, _content, _msg_id, is_bot) = result.unwrap();
    assert!(is_bot);
}

#[test]
fn parse_message_create_missing_fields_returns_none() {
    // Missing author
    let payload = serde_json::json!({
        "id": "123",
        "content": "text",
        "channel_id": "456"
    });
    assert!(DiscordChannel::parse_message_create(&payload).is_none());

    // Missing content
    let payload = serde_json::json!({
        "id": "123",
        "channel_id": "456",
        "author": { "id": "789", "username": "user" }
    });
    assert!(DiscordChannel::parse_message_create(&payload).is_none());
}

#[test]
fn intent_calculation() {
    // GUILDS(1) + GUILD_MESSAGES(512) + DIRECT_MESSAGES(4096) + MESSAGE_CONTENT(32768)
    assert_eq!(DISCORD_INTENTS, 37377);
    assert_eq!(DiscordChannel::intents(), 37377);

    // Verify individual bits
    assert_eq!(DISCORD_INTENTS & (1 << 0), 1);       // GUILDS
    assert_eq!(DISCORD_INTENTS & (1 << 9), 512);     // GUILD_MESSAGES
    assert_eq!(DISCORD_INTENTS & (1 << 12), 4096);   // DIRECT_MESSAGES
    assert_eq!(DISCORD_INTENTS & (1 << 15), 32768);  // MESSAGE_CONTENT
}

#[test]
fn allows_sender_empty_list_denies_all() {
    // Default-deny: an unconfigured allowlist refuses everyone;
    // "*" is the explicit allow-everyone opt-in.
    let ch = DiscordChannel::new("token".into(), vec![]);
    assert!(!ch.allows_sender("anyone"));
}

#[test]
fn allows_sender_wildcard() {
    let ch = DiscordChannel::new("token".into(), vec!["*".into()]);
    assert!(ch.allows_sender("111222333"));
}

#[test]
fn allows_sender_specific_ids() {
    let ch = DiscordChannel::new("token".into(), vec!["111".into(), "222".into()]);
    assert!(ch.allows_sender("111"));
    assert!(ch.allows_sender("222"));
    assert!(!ch.allows_sender("333"));
}

#[test]
fn supports_streaming() {
    let ch = DiscordChannel::new("token".into(), vec![]);
    assert!(ch.supports_streaming());
}
